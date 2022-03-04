import torch
from tqdm import tqdm
from util import get_new_sentence
from torch.utils.data import Dataset


def get_rev_txtid(new_sentence, w2id, max_seq,k):

    ant_txtid=[[] for _ in range(max_seq)]
    if len(new_sentence) > max_seq:
        new_sentence = new_sentence[:max_seq]
    while len(new_sentence) < max_seq:
        new_sentence.append(['[PAD]'])
        
    for index,word_lis in enumerate(new_sentence):
        for token in word_lis:
            if token in w2id:
                ant_txtid[index].append(w2id[token])
            else:
                ant_txtid[index].append(w2id['[UNK]'])
        while (len(ant_txtid[index]) < k):
            ant_txtid[index].append(w2id['[MUL]'])

    return ant_txtid

def get_raw_txtid(txt,w2id,max_seq):
    txtid= []

    for i in (txt.split()):
        if i in w2id:
            txtid.append(w2id[i])
        else:
            txtid.append(w2id['[UNK]'])

    while(len(txtid)<max_seq):
        txtid.append(w2id['[PAD]'])
    if (len(txtid)>max_seq):
        txtid=txtid[:max_seq]

    return txtid

def get_data(path, w2id, ant_word_dic,max_seq,k):
    src_text,rev_text, labels = [], [], []

    with open(path, encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip()
            label = int(line[0])
            labels.append(label)
            raw_text = line[2:]
            raw_text_id=get_raw_txtid(raw_text,w2id,max_seq)
            src_text.append(raw_text_id)
            new_sentence = get_new_sentence(raw_text, ant_word_dic)  # list:[[word],[word0，word1]...]
            rev_text_id = get_rev_txtid(new_sentence, w2id, max_seq,k)
            rev_text.append(rev_text_id)
    return src_text, rev_text, labels


class textDataset_genetate(Dataset):
    def __init__(self, path, w2id, ant_word_dic,max_seq,k):
        src_text, tgt_text, label = get_data(path, w2id,ant_word_dic,max_seq,k)
        self.src = torch.tensor(src_text)  # batch seq
        self.tgt = torch.tensor(tgt_text)
        self.label = torch.tensor(label)

    def __getitem__(self, i):
        return (self.src[i], self.tgt[i], self.label[i])

    def __len__(self):
        return self.src.size(0)



