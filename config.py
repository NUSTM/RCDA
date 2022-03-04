import torch


class Config(object):

    def __init__(self):
        self.data_type = 'sst5'
        self.num_labels = 5 if self.data_type=='sst5' or self.data_type=='yelp' else 2
        self.m = 32
        self.k=3
        self.t = 0.8 if self.num_labels == 2 else 0.5

        self.max_seqlen=50
        self.hid_size = 300
        self.batch_size = 16
        self.rl_batch_size= 8
        self.emb_dim = 300
        self.lr=1e-3
        self.dropout=0.3
        self.device =torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.pretrained_cls_epoch = 80
        self.pretrained_lm_epoch = 40
        self.rl_lm_epoch = 60

        self.sc_max_acc=0
        self.val_max_acc=0
        self.max_acc=0
        self.log_step = 100
        self.sample_step = 1000

        self.rev_pred_path='./data/'+self.data_type+'/rev.pkl'
        self.cls_save_path='./data/'+self.data_type+'/raw.pkl'
        self.vocab_path='./data/'+self.data_type+'/vocab.txt'
        self.lm_path='./data/'+self.data_type+'/lm.pkl'
        self.ant_dataset='./data/'+self.data_type+'/ant_dataset.pkl'
        self.syn_dataset='./data/'+self.data_type+'/syn_dataset.pkl'

        self.train_path,self.val_path,self.test_path,self.ant_dic_path,self.syn_dic_path,\
            self.new_train_path,self.new_val_path,self.new_test_path,self.sc_save_path=self.get_data_path()
        self.data_path=[self.train_path,self.val_path,self.test_path]

    def get_data_path(self):
        if self.data_type=='rt':
            train_path='./data/'+self.data_type+'/rt-polarity.all.train'
            val_path='./data/'+self.data_type+'/rt-polarity.all.dev'
            test_path='./data/'+self.data_type+'/rt-polarity.all.test'
        elif self.data_type=='sst':
            train_path='./data/'+self.data_type+'/stsa.binary.train'
            val_path='./data/'+self.data_type+'/stsa.binary.dev'
            test_path='./data/'+self.data_type+'/stsa.binary.test'

        elif self.data_type=='sst5':
            train_path='./data/'+self.data_type+'/stsa.fine.train'
            val_path='./data/'+self.data_type+'/stsa.fine.dev'
            test_path='./data/'+self.data_type+'/stsa.fine.test'
        elif self.data_type=='yelp':
            train_path='./data/'+self.data_type+'/yelp.train'
            val_path='./data/'+self.data_type+'/yelp.dev'
            test_path='./data/'+self.data_type+'/yelp.test'
            self.pretrained_cls_epoch=20
            self.pretrained_lm_epoch=10
            self.rl_lm_epoch=20
            self.max_seqlen=120
        ant_dic_path='./data/'+self.data_type+'/ant_voc.txt'
        syn_dic_path='./data/'+self.data_type+'/syn_voc.txt'
        new_train_path='./data/'+self.data_type+'/f.train'
        new_val_path='./data/'+self.data_type+'/f.dev'
        new_test_path='./data/'+self.data_type+'/f.test'
        sc_save_path='./data/'+self.data_type+'/model_save/cls_model.pkl'

        return train_path,val_path,test_path,ant_dic_path,syn_dic_path,new_train_path,new_val_path,new_test_path,sc_save_path
