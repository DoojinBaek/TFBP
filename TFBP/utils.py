import os
import csv
import gzip
import math
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

DNAbases='ACGT' # DNA bases

def seqtopad(seq,motif_len):
    rows=len(seq)+2*motif_len-2
    S=np.empty([rows,4])
    base= DNAbases 
    for i in range(rows):
        for j in range(4):
            if i-motif_len+1<len(seq) and seq[i-motif_len+1]=='N' or i<motif_len-1 or i>len(seq)+motif_len-2:
                S[i,j]=np.float32(0.25)
            elif seq[i-motif_len+1]==base[j]:
                S[i,j]=np.float32(1)
            else:
                S[i,j]=np.float32(0)
    return np.transpose(S)

def dinuc_shuffling(seq):
    b=[seq[i:i+2] for i in range(0, len(seq), 2)]
    random.shuffle(b)
    d=''.join([str(x) for x in b])
    return d

def complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    complement_seq = [complement[nt] for nt in seq] # nt stands for nucleotide
    return complement_seq
  
def reverse_complement(seq):
    seq = list(seq)
    seq.reverse()
    return ''.join(complement(seq))

def logsampler(a,b):
        x=np.random.uniform(low=0,high=1)
        y=10**((math.log10(b)-math.log10(a))*x + math.log10(a))
        return y
    
def sqrtsampler(a,b):
        
        x=np.random.uniform(low=0,high=1)
        y=(b-a)*math.sqrt(x)+a
        return y

# utils
def hyperparameters(case_num, code_test, opt_type, scheduler_type, lr_type_adam, lr_type_sgd, dropout_rate_type, pool_type):
    (share, remainder) = divmod(case_num, len(opt_type))
    opt = opt_type[remainder]
    (share, remainder) = divmod(share, len(scheduler_type))
    scheduler = scheduler_type[remainder]
    (share, remainder) = divmod(share, len(lr_type_adam))
    if(opt == 'SGD'):
        lr = lr_type_sgd[remainder]
    else:
        lr = lr_type_adam[remainder]
    (share, remainder) = divmod(share, len(dropout_rate_type))
    dropout_rate = dropout_rate_type[remainder]
    (share, remainder) = divmod(share, len(pool_type))
    pool = pool_type[remainder]

    return opt, scheduler, lr, dropout_rate, pool

def write_settings(name, id, case_num, total_cases, pool, dropout_rate, lr, scheduler, opt):
    with open("./results/"+name+'-'+id+'/'+'experiemt/'+str(case_num)+'.txt', "a") as file:
        file.write("---"*35)
        file.write("\n")
        file.write('TF : ')
        file.write(name)
        file.write('\n')
        file.write(str(case_num+1))
        file.write('th experiment over ')
        file.write(str(total_cases))
        file.write('\n')
        file.write('pool : ')
        file.write(str(pool))
        file.write(', dropout rate : ')
        file.write(str(dropout_rate))
        file.write(', lr : ')
        file.write(str(lr))
        file.write(', scheduler : ')
        file.write(str(scheduler))
        file.write(', optimizer : ')
        file.write(str(opt))
        file.write('\n')
    file.close()

def write_settings_test(name, id, pool, dropout_rate, lr, scheduler, opt):
    with open("./results/"+name+'-'+id+'/'+'test/'+'setting'+'.txt', "a") as file:
        file.write("---"*35)
        file.write("\n")
        file.write('TF : ')
        file.write(name)
        file.write('\n')
        file.write('pool : ')
        file.write(str(pool))
        file.write(', dropout rate : ')
        file.write(str(dropout_rate))
        file.write(', lr : ')
        file.write(str(lr))
        file.write(', scheduler : ')
        file.write(str(scheduler))
        file.write(', optimizer : ')
        file.write(str(opt))
        file.write('\n')
    file.close()


def mkdir(name, id):
    if not os.path.exists("./results/"+name+'-'+id+'/'+'experiemt/'):
        os.makedirs("./results/"+name+'-'+id+'/'+'experiemt/')
    if not os.path.exists("./results/"+name+'-'+id+'/'+'train/'):
        os.makedirs("./results/"+name+'-'+id+'/'+'train/')
    if not os.path.exists("./results/"+name+'-'+id+'/'+'valid/'):
        os.makedirs("./results/"+name+'-'+id+'/'+'valid/')
    if not os.path.exists("./results/"+name+'-'+id+'/'+'test/'):
        os.makedirs("./results/"+name+'-'+id+'/'+'test/')

def write_train_valid_result(name, id, case_num, AUC_training, Loss_training_bce, Loss_training_rest, AUC_validation, Loss_validation_bce, Loss_validation_rest, model_num, epoch):
    if (epoch == 0):
        with open("./results/"+name+'-'+id+'/'+'train/'+str(case_num)+'.txt', "a") as file:
            file.write('Model ')
            file.write(str(model_num+1))
            file.write('\n')
        file.close()
        with open("./results/"+name+'-'+id+'/'+'valid/'+str(case_num)+'.txt', "a") as file:
            file.write('Model ')
            file.write(str(model_num))
            file.write('\n')
        file.close() 
    with open("./results/"+name+'-'+id+'/'+'train/'+str(case_num)+'.txt', "a") as file:
        file.write(str(AUC_training))
        file.write(':')
        file.write(str(Loss_training_bce))
        file.write('+')
        file.write(str(Loss_training_rest))
        file.write('\n')
    file.close()
    with open("./results/"+name+'-'+id+'/'+'valid/'+str(case_num)+'.txt', "a") as file:
        file.write(str(AUC_validation))
        file.write(':')
        file.write(str(Loss_validation_bce))
        file.write('+')
        file.write(str(Loss_validation_rest))
        file.write('\n')
    file.close()

def write_train_test_result(name, id, AUC_training, Loss_training_bce, Loss_training_rest, AUC_test, Loss_test_bce, Loss_test_rest, model_num):
    with open("./results/"+name+'-'+id+'/'+'test/'+str(model_num)+'.txt', "a") as file:
        file.write(str(AUC_training))
        file.write(':')
        file.write(str(Loss_training_bce))
        file.write('+')
        file.write(str(Loss_training_rest))
        file.write(' | ')
        file.write(str(AUC_test))
        file.write(':')
        file.write(str(Loss_test_bce))
        file.write('+')
        file.write(str(Loss_test_rest))
        file.write('\n')
    file.close()

def write_test_result(name, id, AUC_test, Loss_test_bce, Loss_test_rest):
    with open("./results/"+name+'-'+id+'/'+'test/'+'performance'+'.txt', "a") as file:
        file.write(str(AUC_test))
        file.write(':')
        file.write(str(float(Loss_test_bce)))
        file.write('+')
        file.write(str(float(Loss_test_rest)))
        file.write('\n')
    file.close()

def find_best_setting(name, id, epoch):
    experiment_list = os.listdir("./results/"+name+'-'+id+'/'+'experiemt/')
    train_list = os.listdir("./results/"+name+'-'+id+'/'+'train/')
    valid_list = os.listdir("./results/"+name+'-'+id+'/'+'valid/')

    settings = {}
    trains = {}
    valids = {}

    for i in range(len(experiment_list)):
        f = open("./results/"+name+'-'+id+'/'+'experiemt/' + experiment_list[i])
        while True:
            line = f.readline()
            if not line: break
            if 'pool' in line:
                settings[i] = line
        f.close()
        f = open("./results/"+name+'-'+id+'/'+'train/' + train_list[i])
        while True:
            line = f.readlines()
            if not line: break
            trains[i] = line
        f.close()
        f = open("./results/"+name+'-'+id+'/'+'valid/' + valid_list[i])
        while True:
            line = f.readlines()
            if not line: break
            valids[i] = line
        f.close()

    valid_bests = {}

    for experiment_idx in range(len(experiment_list)):
        model1 = valids[experiment_idx][1:epoch+1]
        model2 = valids[experiment_idx][epoch+2:2*epoch+2]
        model3 = valids[experiment_idx][2*epoch+3:]

        model1_parsed = []
        model2_parsed = []
        model3_parsed = []

        for i in range(epoch):
            model1_parsed.append(float(model1[i].split(':')[0]))
            model2_parsed.append(float(model2[i].split(':')[0]))
            model3_parsed.append(float(model3[i].split(':')[0]))

        auc_mean = []
        for i in range(epoch):
            auc_mean.append((model1_parsed[i] + model2_parsed[i] + model3_parsed[i])/3)

        valid_bests[experiment_idx] = max(auc_mean)
        
    best_setting_idx = 0
    best_auc = 0

    for experiment_idx in range(len(experiment_list)):
        if best_auc < valid_bests[experiment_idx]:
            best_auc = valid_bests[experiment_idx]
            best_setting_idx = experiment_idx

    best_setting = settings[best_setting_idx]
    
    return best_setting
# datasets

def datasets(file_path):
    '''
    Input : path to the datasets

    Output : list of dataset names 

        dataset_names[i][0] for list of AC.seq.gz datasets

        dataset_names[i][1] for list of B.seq.gz datasets
    '''
    path = file_path
    files = os.listdir(path)

    train = []
    test = []
    dataset_names = []

    for file in files:
        if file.endswith("AC.seq.gz"):
            train.append(path+file)
        elif file.endswith("B.seq.gz"):
            test.append(path+file)

    train.sort()
    test.sort()

    if(len(train) != len(test)):
        raise Exception("Dataset Corrputed. Please Download The Dataset Again")

    for i in range(len(train)):
        dataset_names.extend([[train[i], test[i]]])

    return dataset_names

'''
STL
'''
# class Chip():
#     def __init__(self,filename,motif_len=24,reverse_complemet_mode=False):
#         self.file = filename
#         self.motif_len = motif_len
#         self.reverse_complemet_mode=reverse_complemet_mode
            
#     def openFile(self):
#         train_dataset=[]
#         with gzip.open(self.file, 'rt') as data:
#             next(data)
#             reader = csv.reader(data,delimiter='\t')
#             if not self.reverse_complemet_mode:
#               for row in reader:
#                       train_dataset.append([seqtopad(row[2],self.motif_len),[1]])
#                       train_dataset.append([seqtopad(dinuc_shuffling(row[2]),self.motif_len),[0]])
#             else:
#               for row in reader:
#                       train_dataset.append([seqtopad(row[2],self.motif_len),[1]])
#                       train_dataset.append([seqtopad(reverse_complement(row[2]),self.motif_len),[1]])
#                       train_dataset.append([seqtopad(dinuc_shuffling(row[2]),self.motif_len),[0]])
#                       train_dataset.append([seqtopad(dinuc_shuffling(reverse_complement(row[2])),self.motif_len),[0]])


#         size=int(len(train_dataset)/3)

#         random.seed(1127)
#         random.shuffle(train_dataset)

#         valid = train_dataset[:size]

#         net_train = train_dataset[size:]

#         return net_train, valid, train_dataset

# class chipseq_dataset(Dataset):
#     def __init__(self,xy=None):
#         self.x_data=np.asarray([el[0] for el in xy],dtype=np.float32)
#         self.y_data =np.asarray([el[1] for el in xy ],dtype=np.float32)
#         self.x_data = torch.from_numpy(self.x_data)
#         self.y_data = torch.from_numpy(self.y_data)
#         self.len=len(self.x_data)
      
#     def __getitem__(self, index):
#         return self.x_data[index], self.y_data[index]

#     def __len__(self):
#         return self.len

# def dataset_loader(path, batch_size = 64, reverse_mode = False):

#     chipseq=Chip(path, reverse_complemet_mode=reverse_mode)

#     train, valid, all=chipseq.openFile()

#     train_dataset=chipseq_dataset(train)
#     valid_dataset=chipseq_dataset(valid)
#     all_dataset=chipseq_dataset(all)

#     batchSize=batch_size

#     if reverse_mode:
#         train_loader = DataLoader(dataset=train_dataset,batch_size=batchSize,shuffle=False)
#         valid_loader = DataLoader(dataset=valid_dataset,batch_size=batchSize,shuffle=False)
#         all_loader=DataLoader(dataset=all_dataset,batch_size=batchSize,shuffle=False)
#     else:
#         train_loader = DataLoader(dataset=train_dataset,batch_size=batchSize,shuffle=True)
#         valid_loader = DataLoader(dataset=valid_dataset,batch_size=batchSize,shuffle=False)
#         all_loader=DataLoader(dataset=all_dataset,batch_size=batchSize,shuffle=False)

#     return train_loader, valid_loader, all_loader

# class Chip_test():
#     def __init__(self,filename,motif_len,reverse_complemet_mode=False):
#         self.file = filename
#         self.motif_len = motif_len
#         self.reverse_complemet_mode=reverse_complemet_mode
            
#     def openFile(self):
#         test_dataset=[]
#         with gzip.open(self.file, 'rt') as data:
#             next(data)
#             reader = csv.reader(data,delimiter='\t')
#             if not self.reverse_complemet_mode:
#               for row in reader:
#                       test_dataset.append([seqtopad(row[2],self.motif_len),[int(row[3])]])
#             else:
#               for row in reader:
#                       test_dataset.append([seqtopad(row[2],self.motif_len),[int(row[3])]])
#                       test_dataset.append([seqtopad(reverse_complement(row[2]),self.motif_len),[int(row[3])]])
            
#         return test_dataset

# def test_dataset_loader(filepath, motif_len):
#     chipseq_test=Chip_test(filepath, motif_len)
#     test_data=chipseq_test.openFile()

#     test_dataset=chipseq_dataset(test_data)
#     batchSize=test_dataset.__len__() # at once

#     test_loader = DataLoader(dataset=test_dataset,batch_size=batchSize,shuffle=False)

#     return test_loader

'''
STL_Like_DeepBind
'''
class Chip():
    def __init__(self,filename,motif_len=24):
        self.file = filename
        self.motif_len = motif_len
            
    def openFile(self):
        train_dataset=[]
        with gzip.open(self.file, 'rt') as data:
            next(data)
            reader = csv.reader(data,delimiter='\t')

            for row in reader:
                train_dataset.append([seqtopad(row[2],self.motif_len),[1]])
                train_dataset.append([seqtopad(dinuc_shuffling(row[2]),self.motif_len),[0]])

        size=int(len(train_dataset)/3)

        random.seed(1127)
        random.shuffle(train_dataset)

        valid1 = train_dataset[:size]
        valid2 = train_dataset[size:2*size]
        valid3 = train_dataset[2*size:]

        train1 = valid2 + valid3
        train2 = valid1 + valid2
        train3 = valid1 + valid3

        return train1, train2, train3, valid1, valid2, valid3, train_dataset

class chipseq_dataset(Dataset):
    def __init__(self,xy=None):
        self.x_data=np.asarray([el[0] for el in xy],dtype=np.float32)
        self.y_data =np.asarray([el[1] for el in xy ],dtype=np.float32)
        self.x_data = torch.from_numpy(self.x_data)
        self.y_data = torch.from_numpy(self.y_data)
        self.len=len(self.x_data)
      
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def dataset_loader(path, motif_len, batch_size = 64):

    chipseq=Chip(path, motif_len)

    train1, train2, train3, valid1, valid2, valid3, all = chipseq.openFile()

    train_dataset_1 = chipseq_dataset(train1)
    valid_dataset_1 = chipseq_dataset(valid1)
    train_dataset_2 = chipseq_dataset(train2)
    valid_dataset_2 = chipseq_dataset(valid2)
    train_dataset_3 = chipseq_dataset(train3)
    valid_dataset_3 = chipseq_dataset(valid3)
    all_dataset=chipseq_dataset(all)

    batchSize=batch_size

    train_loader_1 = DataLoader(dataset=train_dataset_1,batch_size=batchSize,shuffle=False)
    valid_loader_1 = DataLoader(dataset=valid_dataset_1,batch_size=batchSize,shuffle=False)
    train_loader_2 = DataLoader(dataset=train_dataset_2,batch_size=batchSize,shuffle=False)
    valid_loader_2 = DataLoader(dataset=valid_dataset_2,batch_size=batchSize,shuffle=False)
    train_loader_3 = DataLoader(dataset=train_dataset_3,batch_size=batchSize,shuffle=False)
    valid_loader_3 = DataLoader(dataset=valid_dataset_3,batch_size=batchSize,shuffle=False)
    all_loader=DataLoader(dataset=all_dataset,batch_size=batchSize,shuffle=False)


    return train_loader_1, valid_loader_1, train_loader_2, valid_loader_2, train_loader_3, valid_loader_3, all_loader

class Chip_test():
    def __init__(self,filename,motif_len):
        self.file = filename
        self.motif_len = motif_len
            
    def openFile(self):
        test_dataset=[]
        with gzip.open(self.file, 'rt') as data:
            next(data)
            reader = csv.reader(data,delimiter='\t')

            for row in reader:
                test_dataset.append([seqtopad(row[2],self.motif_len),[int(row[3])]])
       
        return test_dataset

def test_dataset_loader(filepath, motif_len):
    chipseq_test=Chip_test(filepath, motif_len)
    test_data=chipseq_test.openFile()

    test_dataset=chipseq_dataset(test_data)
    batchSize=test_dataset.__len__() # at once

    test_loader = DataLoader(dataset=test_dataset,batch_size=batchSize,shuffle=False)

    return test_loader

'''
MTL
'''
class Chip_MT():
    def __init__(self, filename1, filename2, motif_len=24, reverse_complemet_mode=False):
        self.file1 = filename1
        self.file2 = filename2
        self.motif_len = motif_len
        self.reverse_complemet_mode=reverse_complemet_mode
            
    def openFile(self):
        train_dataset=[]

        with gzip.open(self.file1, 'rt') as data:
            next(data)
            reader = csv.reader(data,delimiter='\t')
            for row in reader:
                train_dataset.append([seqtopad(row[2],self.motif_len),[1], [0]])
                train_dataset.append([seqtopad(dinuc_shuffling(row[2]),self.motif_len),[0], [0]])

        with gzip.open(self.file2, 'rt') as data:
            next(data)
            reader = csv.reader(data,delimiter='\t')
            for row in reader:
                train_dataset.append([seqtopad(row[2],self.motif_len),[1], [1]])
                train_dataset.append([seqtopad(dinuc_shuffling(row[2]),self.motif_len),[0], [1]])

        size=int(len(train_dataset)/3)

        random.seed(1127)
        random.shuffle(train_dataset)

        valid = train_dataset[:size]

        net_train = train_dataset[size:]

        return net_train, valid, train_dataset

class chipseq_dataset_MT(Dataset):
    def __init__(self,xy=None):
        self.x_data=np.asarray([el[0] for el in xy],dtype=np.float32)
        self.y_data =np.asarray([el[1] for el in xy ],dtype=np.float32)
        self.z_data = np.asarray([el[2] for el in xy], dtype=np.int32)
        self.x_data = torch.from_numpy(self.x_data)
        self.y_data = torch.from_numpy(self.y_data)
        self.z_data = torch.from_numpy(self.z_data)
        self.len=len(self.x_data)
      
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.z_data[index]

    def __len__(self):
        return self.len

def dataset_loader_MT(path1, path2, batch_size = 64, reverse_mode = False):

    chipseq=Chip_MT(path1, path2, reverse_complemet_mode=reverse_mode)

    train, valid, all=chipseq.openFile()

    train_dataset=chipseq_dataset_MT(train)
    valid_dataset=chipseq_dataset_MT(valid)
    all_dataset=chipseq_dataset_MT(all)

    batchSize=batch_size

    if reverse_mode:
        train_loader = DataLoader(dataset=train_dataset,batch_size=batchSize,shuffle=False)
        valid_loader = DataLoader(dataset=valid_dataset,batch_size=batchSize,shuffle=False)
        all_loader=DataLoader(dataset=all_dataset,batch_size=batchSize,shuffle=False)
    else:
        train_loader = DataLoader(dataset=train_dataset,batch_size=batchSize,shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset,batch_size=batchSize,shuffle=False)
        all_loader=DataLoader(dataset=all_dataset,batch_size=batchSize,shuffle=False)

    return train_loader, valid_loader, all_loader

class Chip_test_MT():
    def __init__(self, filename1, filename2, motif_len,reverse_complemet_mode=False):
        self.file1 = filename1
        self.file2 = filename2
        self.motif_len = motif_len
        self.reverse_complemet_mode=reverse_complemet_mode
            
    def openFile(self):
        test_dataset=[]

        with gzip.open(self.file1, 'rt') as data:
            next(data)
            reader = csv.reader(data,delimiter='\t')
            for row in reader:
                    test_dataset.append([seqtopad(row[2],self.motif_len),[int(row[3])], [0]])

        with gzip.open(self.file2, 'rt') as data:
            next(data)
            reader = csv.reader(data,delimiter='\t')
            for row in reader:
                    test_dataset.append([seqtopad(row[2],self.motif_len),[int(row[3])], [1]])
        
        return test_dataset

def test_dataset_loader_MT(path1, path2, motif_len):
    chipseq_test=Chip_test_MT(path1, path2, motif_len)
    test_data=chipseq_test.openFile()

    test_dataset=chipseq_dataset_MT(test_data)
    batchSize=test_dataset.__len__() # at once

    test_loader = DataLoader(dataset=test_dataset,batch_size=batchSize,shuffle=False)

    return test_loader
