'''
pretrain LM_model
'''
import torch
import numpy as np
from tqdm import tqdm
from config import Config



config = Config()


def get_confid(model_senti, bert_id):
    confid = torch.softmax(model_senti(bert_id),-1)
    return confid


def lm_train(model_lm, loader, opt,vocab_size):
    min_loss = 99999999
    for epoch in range(1, 1 + config.pretrained_lm_epoch):
        all_loss = 0
        print('lm_pretrain [epoch:{}] begin training...'.format(epoch))
        for src,tgt, _ in tqdm(loader):
            temp=torch.zeros(src.shape[0],src.shape[1],vocab_size)
            for i in range(temp.shape[0]):
                for j in range(temp.shape[1]):
                    for id in tgt[i][j]:
                        if id==2:
                            break
                        temp[i,j,id]=1
            tgt=temp
            src,tgt=src.to(config.device), tgt.to(config.device)
            opt.zero_grad()
            loss, predict= model_lm(src, labels=tgt)
            loss = loss.sum()
            all_loss += loss.item()
            loss.backward()
            opt.step()
        print('\r epoch[{}] finished - loss: {:.6f}'.format(epoch, all_loss))
        if all_loss < min_loss:
            min_loss = all_loss
            torch.save(model_lm, config.lm_path)


