
import nltk
from tqdm import tqdm
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


ADJ = ['JJ' ,'JJS' ,'JJR']#形容词
RB = ['RB' , 'RBS' , 'RBR'] #副词
VERB = ['VB' ,'VBZ' , 'VBD' , 'VBN' , 'VBG' , 'VBP'] #
NOUN=['NN','NNS']

def get_antonym_word_list(word):
    antonym_word_list = []
    for syn in wn.synsets(word):
        for term in syn._lemmas:
            if term.antonyms():
                antonym_word_list.append(term.antonyms()[0].name())
        for sim_syn in syn.similar_tos():
            for term in sim_syn._lemmas:
                if term.antonyms():
                    antonym_word_list.append(term.antonyms()[0].name())
    return list(set(antonym_word_list))

def get_syn_word_list(word):
    syn_word_list = []
    for _, sys in enumerate(wn.synsets(word)):
        for term in sys.lemma_names():
            if word.lower() not in term.lower() and word.lower() not in term.lower() and len(term.split('_'))==1 and len(term.split('-'))==1:
                syn_word_list.append(term)
    return list(set(syn_word_list))

def get_word_dic(data_paths,save_path,vocab_path):
    stop_words = set(stopwords.words('english'))
    frequen={}
    all_vocab=set()
    vocab,syn_vocab=set(),set()
    word_dic={}
    for data_path in data_paths:
        with open(data_path,encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.strip()
                raw_text = line[2:]
                for word,pos in nltk.pos_tag(raw_text.split()):
                    if word in frequen:
                        frequen[word]+=1
                    else:
                        frequen[word]=1
                    all_vocab.add(word)
                    if pos in (ADJ+RB+VERB):
                        vocab.add(word)
                    if pos in (NOUN):
                        syn_vocab.add(word)
    for word in vocab:
        if word in stop_words:
            continue
        word_list=[]
        
        ant_word_list=get_antonym_word_list(word)
        if ant_word_list!=[]:
            ant_word_dic={}
            for ant_word in ant_word_list:
                if ant_word in frequen:
                    ant_word_dic[ant_word]=frequen[ant_word]
                else: 
                    ant_word_dic[ant_word]=0
            result = sorted(ant_word_dic.items(), key = lambda x :(-x[1]))
            num=0
            for tup in result:
                if(num>=1):
                    break
                word_list.append(tup[0])
                num+=1

        else:
            syn_word_list = get_syn_word_list(word)
            syn_word_dic={}
            for syn_word in syn_word_list:
                if syn_word in frequen:
                    syn_word_dic[syn_word]=frequen[syn_word]
                else: 
                    syn_word_dic[syn_word]=0
            result = sorted(syn_word_dic.items(), key = lambda x :(-x[1]))
            num=0
            for tup in result:
                if(num>=1):
                    break
                word_list.append(tup[0])
                num+=1

        if word_list==[]:
            continue
        for new_word in word_list:
            all_vocab.add(new_word)
            if word not in word_dic:
                word_dic[word]=[new_word]
            else:
                word_dic[word].append(new_word)

    for word in syn_vocab:
        if word in word_dic:
            continue
        if word in stop_words:
            continue
        word_list=[]
        syn_word_list = get_syn_word_list(word)
        syn_word_dic={}
        for syn_word in syn_word_list:
            if syn_word in frequen:
                syn_word_dic[syn_word]=frequen[syn_word]
            else: 
                syn_word_dic[syn_word]=0
        result = sorted(syn_word_dic.items(), key = lambda x :(-x[1]))
        num=0
        for tup in result:
            if(num>=1):
                break
            word_list.append(tup[0])
            num+=1
        if word_list==[]:
            continue
        for new_word in word_list:
            all_vocab.add(new_word)
            if word not in word_dic:
                word_dic[word]=[new_word]
            else:
                word_dic[word].append(new_word)

    with open(save_path, 'w+',encoding='utf-8') as f:
        for word in word_dic:
            f.write(word+' '+','.join(word_dic[word])+'\n')
    with open(vocab_path, 'w+',encoding='utf-8') as f:
        for word in all_vocab:
            f.write(word+'\n')
 

def load_word_dic(dic_path):
    word_dic={}
    with open(dic_path,encoding='utf-8') as f:
        for line in f:
            word,new_word = line.strip().split()
            word_dic[word]=new_word.split(',')
    return  word_dic

def load_vocab(vocab_path):
    w2id,id2w = {'[PAD]':0,'[UNK]':1,'[MUL]':2},{0:'[PAD]',1:'[UNK]',2:'[MUL]'}
    index = 0
    with open(vocab_path, encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            w2id[word] = index+3
            id2w[index+3]=word
            index+=1
    return w2id,id2w,index+3

def get_new_sentence(sentence,word_dic):
    sentence=sentence.split()
    new_sentence=[]
    for word in sentence:
        if word in word_dic:
            new_sentence.append(word_dic[word])
        else:
            new_sentence.append([word])
    return new_sentence

def token_2_sentence(id_list,id2w):
    sentence_list=[]
    for index in id_list:
        if(index==0):
            break
        sentence_list.append(id2w[index])
    return ' '.join(sentence_list)

