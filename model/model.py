import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
from transformers import BertModel, BertConfig, BertForMaskedLM

class rl_Loss(nn.Module):
    def __init__(self):
        super(rl_Loss, self).__init__()

    def forward(self, prob, target, reward):
        """
        Args:
            N=batch*seq_len
            prob: (batch,seq, vocab)
            target : (batch,seq )
            reward : (batch,seq )
        """
        # print(prob.size())
        # print(target.size())
        # print(reward.size())
        reward=reward.unsqueeze(-1).repeat(1,target.size(-1))
        V = prob.size(-1)
        prob = prob.view(-1, V)
        target = target.view(-1)
        reward = reward.view(-1)
        N = target.size(0)
        one_hot = torch.zeros((N, V))
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1, 1)), 1)
        one_hot = one_hot.bool()
        one_hot = Variable(one_hot)
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        loss = torch.masked_select(prob, one_hot)
        loss = loss * reward
        loss = -torch.sum(loss)
        return loss

    # def backward(self, retain_graph=True):
    #     self.loss.backward(retain_graph=retain_graph)
    #     return self.loss


class lstm_generator(nn.Module):
    def __init__(self,vocab_size,embed_dim,hiden_size):
        super(lstm_generator, self).__init__()
        self.embed=nn.Embedding(vocab_size,embed_dim)
        self.e_drop=nn.Dropout(0.3)
        self.lstm=nn.LSTM(embed_dim,hiden_size,bidirectional=True,batch_first=True)
        self.fc=nn.Linear(hiden_size*2,vocab_size)

    def forward(self, x, labels=None):
        embed=self.e_drop(self.embed(x))
        predict=self.fc(self.lstm(embed)[0])
        if labels is not None:
            return binary_cross_entropy_with_logits(predict, labels), predict
        else:
            return predict

    def sample(self, src, mask_matrix, pad_matrix):
        # print(torch.tensor(tgt).shape)
        output = self.forward(src)  # batch seq vocab
        output = output.masked_fill(mask_matrix, -float('inf')).softmax(-1)
        sample = []
        for i in range(output.shape[1]):
            sample.append(output[:, i, :].multinomial(1, replacement=True))
        sample = torch.cat(sample, dim=1)
        sample = sample.masked_fill(pad_matrix, 0)
      
        return sample
    
    def get_text(self, src, mask_matrix, pad_matrix):
        # print(torch.tensor(tgt).shape)
        output = self.forward(src)  # batch seq vocab
        output = output.masked_fill(mask_matrix, -float('inf'))
        
        new_id = output.argmax(-1)
      
        return new_id



class raw_rnn_cls(nn.Module):
    def __init__(self,vob_size,emb_size,hid_size,num_labels):
        super(raw_rnn_cls, self).__init__()
        self.embed = nn.Embedding(vob_size,emb_size)
        self.lstm = nn.LSTM(emb_size, hid_size,
                            bidirectional=False,
                            batch_first=True)
        self.fc_out = nn.Linear(hid_size , num_labels)
        self.embed_drop = nn.Dropout(0.4)
        self.lstm_drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.embed_drop(self.embed(x))
        _, (h,_) = self.lstm(x)
        h=self.lstm_drop(h.squeeze(0))
        output = self.fc_out(h)
        return output,h

