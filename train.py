import os
import nltk
import torch
import pickle
from tqdm import tqdm
from config import Config
import torch.nn.functional as F
import torch.optim as optim
from train_cls import train_classification
from train_generate import lm_train
from model.model import lstm_generator, raw_rnn_cls
from torch.utils.data import DataLoader
from dataprocess import textDataset_genetate
from util import get_word_dic, load_word_dic, load_vocab

config = Config()


def get_loader(data_paths, w2id, ant_word_dic):
    path=config.ant_dataset

    if not os.path.exists(path):
        datasets = []
        for data_path in data_paths:
            dataset = textDataset_genetate(data_path, w2id,ant_word_dic, config.max_seqlen)
            datasets.append(dataset)
        f = open(path, 'wb')
        pickle.dump(datasets, f)
        f.close()

    f = open(path, 'rb')
    datasets = pickle.load(f)
    f.close()

    train_loader = DataLoader(datasets[0], batch_size=config.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(datasets[1], batch_size=config.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(datasets[2], batch_size=config.batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader

def final_test(model,pres_rev,test_loader,threshold):
    labels,pres_raw=[],[]

    for text,_ ,label in test_loader:
        text ,label = text.to(config.device),label.to(config.device)
        predict,_= model(text)
        pres_raw.extend(predict.tolist())
        labels.extend(label.tolist())

    pres_raw,pres_rev,labels=torch.tensor(pres_raw).cuda(),torch.tensor(pres_rev).cuda(),torch.tensor(labels).cuda()

    for index in range(len(pres_rev)):
        if pres_raw[index].max() < threshold and pres_raw[index].max() < pres_rev[index].max():
            pres_raw[index] = pres_rev[index]

    correct = (torch.max(pres_raw, 1)[1] == labels).sum().item()
    correct_rate = correct / len(pres_raw)
    print(' the max correct rate is {} '.format(correct_rate))

def get_word_frequency(path,num_labels,w2id):
    raw_frequency={}
    stop_words = nltk.corpus.stopwords.words('english')
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            label,raw_text,_=line.split('   ')
            label=int(label)
            for word in raw_text.split():
                if word in w2id and word not in stop_words:
                    if w2id[word] not in raw_frequency:
                        raw_frequency[w2id[word]]=[0]*num_labels
                    
                    raw_frequency[w2id[word]][label]+=1

    for key in raw_frequency:
        raw_frequency[key]=max(raw_frequency[key])/sum(raw_frequency[key])
    
    return raw_frequency

def reward_loss(prediction,label,reward):

    temp = torch.zeros(prediction.shape)
    temp=temp.to(config.device)
    label=temp.scatter(-1,label.unsqueeze(-1),1)
    reward=reward.view(-1,1)
    reward=reward.repeat(1,config.max_seqlen)
    loss=(-label * (prediction + 1e-7).log_softmax(-1)).sum(-1) #batch*seq
    loss=(loss*reward).sum()/label.shape[0]
    return loss

def ant_rl_train(lm_model, ori_model,rev_model,opt_rev_cls, sample_nums, opt,vocab_size, train_loader,val_loader,test_loader):

    step=0
    for epoch in range(1, 1 + config.rl_lm_epoch):
        print('rl_pretrain [epoch:{}] begin training...'.format(epoch))
        for src, rev_tgt, label in tqdm(train_loader):
            step+=1
            src,label = src.to(config.device),label.to(config.device)
            one_hot = torch.zeros(src.shape[0],config.num_labels).to(config.device)
            one_hot = one_hot.scatter(1,label.unsqueeze(1),1).float()

            #####搜索mask matrix
            mask_matrix=torch.ones(src.shape[0],src.shape[1],vocab_size)
            if src.is_cuda:
                mask_matrix=mask_matrix.cuda()
            for i in range(mask_matrix.shape[0]):
                for j in range(mask_matrix.shape[1]):
                    for index in rev_tgt[i][j]:
                        if index==2:   #去掉多标签的pad
                            break
                        mask_matrix[i,j,index]=0
            mask_matrix=mask_matrix.bool()

            pad_matrix=torch.zeros(src.shape[0],src.shape[1])
            if src.is_cuda:
                pad_matrix=pad_matrix.cuda()
            for i in range(pad_matrix.shape[0]):
                for j in range(pad_matrix.shape[1]):
                    if src[i,j]==0:
                        pad_matrix[i,j]=1
            pad_matrix=pad_matrix.bool()

            tgt, new_confids = [], []
            predict=lm_model(src)

            for i in range(sample_nums):
                temp = lm_model.sample(src,mask_matrix,pad_matrix)  # batch seq
                tgt.append(temp.detach())
                if config.val_max_acc>0.75 and config.num_labels==2 or config.val_max_acc>0.35 and config.num_labels==5:
                    new_confids.append(
                        torch.abs(rev_model(temp)[0].softmax(-1).detach() * one_hot).sum(-1).unsqueeze(-1))
                else:
                    new_confids.append(
                        torch.abs(ori_model(temp)[0].softmax(-1).detach() - one_hot).sum(-1).unsqueeze(-1))  # sample_num *batch*1
            avg_confid = torch.cat(new_confids, -1).mean(-1)  #batch

            #training genenrator
            opt.zero_grad()
            loss_all=0
            for i in range(sample_nums):
                reward=new_confids[i].squeeze(-1)-avg_confid
                reward=reward.clamp(min=0)
                loss=reward_loss(predict, tgt[i],reward)
                loss_all += loss
            loss_all.backward()
            opt.step()

            # training rev classification
            opt_rev_cls.zero_grad()
            rev_text=lm_model.get_text(src,mask_matrix,pad_matrix)
            predict,_= rev_model(rev_text)
            loss = F.cross_entropy(predict, label)
            loss.backward()
            opt_rev_cls.step()
            if config.data_type=='yelp' and step % config.sample_step == 0:
                val(lm_model,rev_model,vocab_size, val_loader,test_loader)
        if config.data_type!='yelp':
            val(lm_model,rev_model,vocab_size,val_loader,test_loader)
           

def val(lm_model,rev_model,vocab_size,val_loader,test_loader):
    rev_model = rev_model.eval()
    correct = 0

    for src, rev_tgt, label in val_loader:
        src,label = src.to(config.device),label.to(config.device)

        #####搜索mask matrix
        mask_matrix=torch.ones(src.shape[0],src.shape[1],vocab_size)
        if src.is_cuda:
            mask_matrix=mask_matrix.cuda()
        for i in range(mask_matrix.shape[0]):
            for j in range(mask_matrix.shape[1]):
                for index in rev_tgt[i][j]:
                    if index==2:   #去掉多标签的pad
                        break
                    mask_matrix[i,j,index]=0
        mask_matrix=mask_matrix.bool()

        pad_matrix=torch.zeros(src.shape[0],src.shape[1])
        if src.is_cuda:
            pad_matrix=pad_matrix.cuda()
        for i in range(pad_matrix.shape[0]):
            for j in range(pad_matrix.shape[1]):
                if src[i,j]==0:
                    pad_matrix[i,j]=1

        pad_matrix=pad_matrix.bool()
        rev_text=lm_model.get_text(src,mask_matrix,pad_matrix)

        predict,_= rev_model(rev_text)
        correct += (torch.max(predict, 1)[1] == label).sum()

    acc = 100 * float(correct) / float(len(val_loader.dataset))
    print('Acc of the val set is {}'.format(acc))
    if acc > config.val_max_acc:
        config.val_max_acc = acc
        print('The highest acc of the val set is {}'.format(acc))
        test(lm_model,rev_model,vocab_size,test_loader)
    rev_model = rev_model.train()

def test(lm_model,rev_model,vocab_size,test_loader):
    pres_rev=[]
    rev_model = rev_model.eval()
    correct = 0
    for src,rev_tgt, label in test_loader:
        src,label = src.to(config.device),label.to(config.device)

        #####搜索mask matrix
        mask_matrix=torch.ones(src.shape[0],src.shape[1],vocab_size)
        if src.is_cuda:
            mask_matrix=mask_matrix.cuda()
        for i in range(mask_matrix.shape[0]):
            for j in range(mask_matrix.shape[1]):
                for index in rev_tgt[i][j]:
                    if index==2:   #去掉多标签的pad
                        break
                    mask_matrix[i,j,index]=0
        mask_matrix=mask_matrix.bool()

        pad_matrix=torch.zeros(src.shape[0],src.shape[1])
        if src.is_cuda:
            pad_matrix=pad_matrix.cuda()
        for i in range(pad_matrix.shape[0]):
            for j in range(pad_matrix.shape[1]):
                if src[i,j]==0:
                    pad_matrix[i,j]=1

        pad_matrix=pad_matrix.bool()
        rev_text=lm_model.get_text(src,mask_matrix,pad_matrix)

        predict,_= rev_model(rev_text)
        pres_rev.extend(predict.tolist())
        correct += (torch.max(predict, 1)[1] == label).sum()

    acc = 100 * float(correct) / float(len(test_loader.dataset))
    print('The acc of the test is {}'.format(acc))
    if acc > config.max_acc:
        config.max_acc = acc
        print('The rev_prediction is saved and the acc is {}'.format(acc))
        pickle.dump(pres_rev,open(config.rev_pred_path, 'wb'))

    rev_model = rev_model.train()


def train():

    if not os.path.exists(config.ant_dic_path) or not os.path.exists(config.vocab_path):
        print('get antonym and synonym from wordnet...')
        get_word_dic([config.train_path], config.ant_dic_path,config.vocab_path)

    ant_dic = load_word_dic(config.ant_dic_path)
    w2id,id2w,vocab_size=load_vocab(config.vocab_path)

    print('vocab is loaded, and the vocab_size is :',vocab_size)
    rev_train_loader, rev_val_loader, rev_test_loader = get_loader(
        [config.train_path, config.val_path, config.test_path], w2id,ant_dic)

    ######training raw classification
    if not os.path.exists(config.cls_save_path):
        model=raw_rnn_cls(len(w2id),config.emb_dim,config.hid_size,config.num_labels).to(config.device)
        optimizer = optim.Adam(model.parameters(), config.lr)
        print('begin train raw_cls...')
        train_classification(model, rev_train_loader, rev_val_loader, rev_test_loader, optimizer)

    ######training generator
    config.max_acc=0
    config.val_max_acc=0
    if not os.path.exists(config.lm_path):
        lm_model=lstm_generator(vocab_size,config.emb_dim,config.hid_size)
        lm_model.to(config.device)
        opt_pos = optim.Adam(lm_model.parameters(), config.lr)
        lm_train(lm_model,rev_train_loader,opt_pos,vocab_size)

    ######RL training generator and training rev classification
    if not os.path.exists(config.rev_pred_path):
        lm_model=torch.load(config.lm_path)
        lm_model.train()
        opt_lm = optim.Adam(lm_model.parameters(), config.lr)
        ori_model = torch.load(config.cls_save_path)

        rev_model=raw_rnn_cls(len(w2id),config.emb_dim,config.hid_size,config.num_labels).to(config.device)
        rev_optimizer = optim.Adam(rev_model.parameters(), config.lr)

        ant_rl_train(lm_model, ori_model,rev_model,rev_optimizer, config.m, opt_lm,vocab_size, rev_train_loader, rev_val_loader, rev_test_loader)

    pre_rev=pickle.load(open(config.rev_pred_path, 'rb'))
    model=torch.load(config.cls_save_path)
    final_test(model,pre_rev,rev_test_loader,config.t)

if __name__=='__main__':

    seed=18
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed(seed) #gpu
    torch.backends.cudnn.deterministic=True # cudnn
    train()
