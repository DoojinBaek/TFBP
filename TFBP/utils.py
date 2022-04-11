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

class Chip():
    def __init__(self,filename,motif_len=24,reverse_complemet_mode=False):
        self.file = filename
        self.motif_len = motif_len
        self.reverse_complemet_mode=reverse_complemet_mode
            
    def openFile(self):
        train_dataset=[]
        with gzip.open(self.file, 'rt') as data:
            next(data)
            reader = csv.reader(data,delimiter='\t')
            if not self.reverse_complemet_mode:
              for row in reader:
                      train_dataset.append([seqtopad(row[2],self.motif_len),[1]])
                      train_dataset.append([seqtopad(dinuc_shuffling(row[2]),self.motif_len),[0]])
            else:
              for row in reader:
                      train_dataset.append([seqtopad(row[2],self.motif_len),[1]])
                      train_dataset.append([seqtopad(reverse_complement(row[2]),self.motif_len),[1]])
                      train_dataset.append([seqtopad(dinuc_shuffling(row[2]),self.motif_len),[0]])
                      train_dataset.append([seqtopad(dinuc_shuffling(reverse_complement(row[2])),self.motif_len),[0]])


        size=int(len(train_dataset)/3)

        random.seed(1127)
        random.shuffle(train_dataset)

        valid = train_dataset[:size]

        net_train = train_dataset[size:]

        return net_train, valid, train_dataset

class Chip_two_tfs():
    def __init__(self,filename1, filename2 ,motif_len=24,reverse_complemet_mode=False):
        self.file1 = filename1
        self.file2 = filename2
        self.motif_len = motif_len
        self.reverse_complemet_mode=reverse_complemet_mode
            
    def openFile(self):
        train_dataset=[]

        with gzip.open(self.file1, 'rt') as data:
            next(data)
            reader = csv.reader(data,delimiter='\t')
            if not self.reverse_complemet_mode:
              for row in reader:
                      train_dataset.append([seqtopad(row[2],self.motif_len),[1]])
                      train_dataset.append([seqtopad(dinuc_shuffling(row[2]),self.motif_len),[0]])
            else:
              for row in reader:
                      train_dataset.append([seqtopad(row[2],self.motif_len),[1]])
                      train_dataset.append([seqtopad(reverse_complement(row[2]),self.motif_len),[1]])
                      train_dataset.append([seqtopad(dinuc_shuffling(row[2]),self.motif_len),[0]])
                      train_dataset.append([seqtopad(dinuc_shuffling(reverse_complement(row[2])),self.motif_len),[0]])

        with gzip.open(self.file2, 'rt') as data:
            next(data)
            reader = csv.reader(data,delimiter='\t')
            if not self.reverse_complemet_mode:
              for row in reader:
                      train_dataset.append([seqtopad(row[2],self.motif_len),[1]])
                      train_dataset.append([seqtopad(dinuc_shuffling(row[2]),self.motif_len),[0]])
            else:
              for row in reader:
                      train_dataset.append([seqtopad(row[2],self.motif_len),[1]])
                      train_dataset.append([seqtopad(reverse_complement(row[2]),self.motif_len),[1]])
                      train_dataset.append([seqtopad(dinuc_shuffling(row[2]),self.motif_len),[0]])
                      train_dataset.append([seqtopad(dinuc_shuffling(reverse_complement(row[2])),self.motif_len),[0]])

        size=int(len(train_dataset)/3)

        random.seed(1127)
        random.shuffle(train_dataset)

        valid = train_dataset[:size]

        net_train = train_dataset[size:]

        return net_train, valid, train_dataset

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

def dataset_loader(path, batch_size = 64, reverse_mode = False):

    chipseq=Chip(path, reverse_complemet_mode=reverse_mode)

    train, valid, all=chipseq.openFile()

    train_dataset=chipseq_dataset(train)
    valid_dataset=chipseq_dataset(valid)
    all_dataset=chipseq_dataset(all)

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

def dataset_loader_two_tfs(path1, path2, batch_size = 64, reverse_mode = False):

    chipseq=Chip_two_tfs(path1, path2, reverse_complemet_mode=reverse_mode)

    train, valid, all=chipseq.openFile()

    train_dataset=chipseq_dataset(train)
    valid_dataset=chipseq_dataset(valid)
    all_dataset=chipseq_dataset(all)

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

class Chip_test():
    def __init__(self,filename,motif_len,reverse_complemet_mode=False):
        self.file = filename
        self.motif_len = motif_len
        self.reverse_complemet_mode=reverse_complemet_mode
            
    def openFile(self):
        test_dataset=[]
        with gzip.open(self.file, 'rt') as data:
            next(data)
            reader = csv.reader(data,delimiter='\t')
            if not self.reverse_complemet_mode:
              for row in reader:
                      test_dataset.append([seqtopad(row[2],self.motif_len),[int(row[3])]])
            else:
              for row in reader:
                      test_dataset.append([seqtopad(row[2],self.motif_len),[int(row[3])]])
                      test_dataset.append([seqtopad(reverse_complement(row[2]),self.motif_len),[int(row[3])]])
            
        return test_dataset

def test_dataset_loader(filepath, motif_len):
    chipseq_test=Chip_test(filepath, motif_len)
    test_data=chipseq_test.openFile()

    test_dataset=chipseq_dataset(test_data)
    batchSize=test_dataset.__len__() # at once

    test_loader = DataLoader(dataset=test_dataset,batch_size=batchSize,shuffle=False)

    return test_loader
