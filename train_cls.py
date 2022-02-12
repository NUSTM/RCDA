import torch
import torch.nn.functional as F
from config import Config

config=Config()

def train_classification(model, train_loader, val_loader,test_loader, opt):
    step=0
    config.val_max_acc,config.max_acc=0,0
    epochs=config.pretrained_cls_epoch

    for epoch in range(1, 1 + epochs):
        for text,_ ,label in train_loader:
            step+=1
            text ,label = text.to(config.device),label.to(config.device)
            opt.zero_grad()
            predict,_= model(text)
            loss = F.cross_entropy(predict, label)
            loss.backward()
            opt.step()
            if  step % config.log_step == 0:
                correct = (torch.max(predict, 1)[1] == label).sum()
                acc = 100 * float(correct) / float(config.batch_size)
                print('\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(step,
                                                                               loss,
                                                                               acc,
                                                                               correct,
                                                                               config.batch_size)) 
            if config.data_type=='yelp' and step % config.sample_step == 0:
                val(model, val_loader,test_loader,epoch)
        val(model, val_loader,test_loader,epoch)
    print('The highest acc is {}'.format(config.max_acc))


def val(model, val_loader,test_loader,epoch):
    model = model.eval()
    correct = 0

    for text,_ ,label in val_loader:
        text ,label = text.to(config.device),label.to(config.device)
        predict,_= model(text)
        correct += (torch.max(predict, 1)[1] == label).sum()
    acc = 100 * float(correct) / float(len(val_loader.dataset))
    print('\r epoch[{}]: the acc of val set is: {:.4f}%({}/{}) '.format(epoch,
                                            acc,
                                           correct,
                                           len(val_loader.dataset)))
    if acc > config.val_max_acc:
        config.val_max_acc = acc
        print('The highest acc of the val set is {}'.format(acc))
        test(model,test_loader)

    model = model.train()

def test(model,test_loader):
    model = model.eval()
    correct = 0
    for text,_ ,label in test_loader:
        text ,label = text.to(config.device),label.to(config.device)
        predict,_= model(text)
        correct += (torch.max(predict, 1)[1] == label).sum()
    acc = 100 * float(correct) / float(len(test_loader.dataset))
    print('\r the acc of test set is: {:.4f}%({}/{}) '.format(acc,
                                           correct,
                                           len(test_loader.dataset)))
    if acc > config.max_acc:
        config.max_acc = acc
        print('The model is saved and the acc is {}'.format(acc))
        torch.save(model, config.cls_save_path)
    model = model.train()