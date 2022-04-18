import os
import time
import copy
import datetime

import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F

from sklearn import metrics
from TFBP import datasets, dataset_loader_MT, test_dataset_loader_MT, MTL_Model


if(torch.cuda.is_available()):
    print('Torch',torch.__version__, 'is available')
else:
    print('Torch is not available. Process is terminated')
    quit()

def arg_parser():
    parser = argparse.ArgumentParser(description='Set hyperparameters or you can use default hyperparameter settings defined in the hyperparameter.json file')
    parser.add_argument('--codeTest', type=bool, default=False, help='Do you want to test the code using only one hyperparameters setting?')
    parser.add_argument('--id', type=str, required=True, help='Set the name or id of this experiment')
    args = parser.parse_args()
    return args

def mkdir(name1, name2, id):
    if not os.path.exists("./mtl_results/"+name1+'_'+name2+'-'+id+'/'+'experiemt/'):
        os.makedirs("./mtl_results/"+name1+'_'+name2+'-'+id+'/'+'experiemt/')
    if not os.path.exists("./mtl_results/"+name1+'_'+name2+'-'+id+'/'+'train/'):
        os.makedirs("./mtl_results/"+name1+'_'+name2+'-'+id+'/'+'train/')
    if not os.path.exists("./mtl_results/"+name1+'_'+name2+'-'+id+'/'+'valid/'):
        os.makedirs("./mtl_results/"+name1+'_'+name2+'-'+id+'/'+'valid/')
    if not os.path.exists("./mtl_results/"+name1+'_'+name2+'-'+id+'/'+'test/'):
        os.makedirs("./mtl_results/"+name1+'_'+name2+'-'+id+'/'+'test/')
    return "./mtl_results/"+name1+'_'+name2+'-'+id+'/'

def main():

    start = time.time()
    
    # parsing
    args = arg_parser()
    tfs = ['ARID3A', 'ZBTB7A']
    CodeTesting = args.codeTest
    id = args.id
    print('TF Binding Prediction for', tfs[0], 'and', tfs[1])
    print('Searching for all hyperparameter settings...')

    # Hyperparameters
    num_epochs = 100
    reverse_mode = False
    num_motif_detector = 16
    motif_len = 24
    batch_size = 64
    reg = 2*10**-6
    
    pool_type = ['maxavg']
    dropout_rate_type = [0.3]
    # Learning Rates
    lr_sgd_type = [0.001, 0.005, 0.01, 0.025, 0.05]
    scheduler_type = [True] # use Cosine Annealing or not
    opt_type = ['SGD'] # optimizer
      
    total_cases = len(pool_type)*len(dropout_rate_type)*len(lr_sgd_type)*len(scheduler_type)*len(opt_type)
    print('Total cases :', total_cases)

    # Settings
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # dataset
    path = './data/encode/'
    all_dataset_names = datasets(path)
    TF_to_idx = {'ARID3A' : 0, 'CTCFL' : 1, 'ELK1' : 2, 'FOXA1' : 3, 'GABPA' : 4, 'MYC' : 5, 'REST' : 6, 'SP1' : 7, 'USF1' : 8, 'ZBTB7A' : 9}
    TF1_idx = TF_to_idx[tfs[0]]
    TF2_idx = TF_to_idx[tfs[1]]
    if(CodeTesting):
        print(f'{tfs[0]} idx : {TF1_idx}')
        print(f'{tfs[1]} idx : {TF2_idx}')
        
    tf1_dataset_name = all_dataset_names[TF1_idx]
    tf1_train_dataset_path = tf1_dataset_name[0]
    tf1_test_dataset_path = tf1_dataset_name[1]
    tf1_name = tf1_train_dataset_path.split(path)[1].split("_AC")[0]

    tf2_dataset_name = all_dataset_names[TF2_idx]
    tf2_train_dataset_path = tf2_dataset_name[0]
    tf2_test_dataset_path = tf2_dataset_name[1]
    tf2_name = tf2_train_dataset_path.split(path)[1].split("_AC")[0]

    train_data_loader, valid_data_loader, all_data_loader = dataset_loader_MT(tf1_train_dataset_path, tf2_train_dataset_path, batch_size, reverse_mode)

    for case_num in range(total_cases):
        print("---"*8)
        print(case_num+1, 'th experiment over ', total_cases)

        # specify hyperparameters
        (share, remainder) = divmod(case_num, len(opt_type))
        opt = opt_type[remainder]
        (share, remainder) = divmod(share, len(scheduler_type))
        scheduler = scheduler_type[remainder]
        (share, remainder) = divmod(share, len(lr_sgd_type))
        if(opt == 'SGD'):
            lr = lr_sgd_type[remainder]
        else:
            print('error')
        (share, remainder) = divmod(share, len(dropout_rate_type))
        dropout_rate = dropout_rate_type[remainder]
        (share, remainder) = divmod(share, len(pool_type))
        pool = pool_type[remainder]

        result_path = mkdir(tf1_name, tf2_name, id)
        with open(result_path+'experiemt/'+str(case_num)+'.txt', "a") as file:
            file.write("---"*35)
            file.write("\n")
            file.write('TF : ')
            file.write(tf1_name)
            file.write(', ')
            file.write(tf2_name)
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

        # Model Training

        print('Model Training')

        model = MTL_Model(num_motif_detector,motif_len,pool,'training',lr, dropout_rate, device)

        # optimizer
        if opt == 'SGD':
            optimizer = torch.optim.SGD([
                model.net.wConv1, model.net.wRect1, model.net.wConv2, model.net.wRect2,
                model.net1.wNeu,model.net1.wNeuBias,model.net1.wHidden,model.net1.wHiddenBias,
                model.net2.wNeu,model.net2.wNeuBias,model.net2.wHidden,model.net2.wHiddenBias
            ], lr = lr, momentum = 0.9)
        else:
            optimizer = torch.optim.SGD([
                model.net.wConv1, model.net.wRect1, model.net.wConv2, model.net.wRect2,
                model.net1.wNeu,model.net1.wNeuBias,model.net1.wHidden,model.net1.wHiddenBias,
                model.net2.wNeu,model.net2.wNeuBias,model.net2.wHidden,model.net2.wHiddenBias
            ], lr = lr)

        # scheduler
        if scheduler == True:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
        else:
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1) # constant learning rate

        train_loader = train_data_loader
        valid_loader = valid_data_loader

        loss_best = 1000
        
        for epoch in range(num_epochs):
            
            if epoch%10 == 0:
                print(epoch, 'th epoch over ', num_epochs)
                
            with torch.no_grad():
                # for train set
                model.net.mode = 'test'
                model.net1.mode = 'test'
                model.net2.mode = 'test'
                auc1 = []
                auc2 = []
                train_loss = []
                for idx, (data, target, task) in enumerate(train_loader):
                    data1, data2, target1, target2 = divide_batch(data, target, task, device)

                    # Forward pass
                    output1 = model.forward(data1, 0)
                    output2 = model.forward(data2, 1)

                    pred1_sig = torch.sigmoid(output1)
                    pred2_sig = torch.sigmoid(output2)

                    loss = F.binary_cross_entropy(pred1_sig,target1) + reg*model.net.wConv1.norm() + reg*model.net.wConv2.norm() + reg*model.net1.wHidden.norm() + reg*model.net1.wNeu.norm()
                    loss += F.binary_cross_entropy(pred2_sig,target2) + reg*model.net.wConv1.norm() + reg*model.net.wConv2.norm() + reg*model.net2.wHidden.norm() + reg*model.net2.wNeu.norm()
                    train_loss.append(loss.cpu())

                    pred1 = pred1_sig.cpu().detach().numpy().reshape(output1.shape[0])
                    pred2 = pred2_sig.cpu().detach().numpy().reshape(output2.shape[0])

                    label1 = target1.cpu().numpy().reshape(output1.shape[0])
                    label2 = target2.cpu().numpy().reshape(output2.shape[0])

                    try:
                        auc1.append(metrics.roc_auc_score(label1, pred1))
                        auc2.append(metrics.roc_auc_score(label2, pred2))
                    except ValueError:
                        pass

                AUC_training_1 = np.mean(auc1)
                AUC_training_2 = np.mean(auc2)
                Loss_train = np.mean(train_loss)

                # for valid set
                model.net.mode = 'test'
                model.net1.mode = 'test'
                model.net2.mode = 'test'
                auc1 = []
                auc2 = []
                valid_loss = []
                for idx, (data, target, task) in enumerate(valid_loader):
                    data1, data2, target1, target2 = divide_batch(data, target, task, device)

                    # Forward pass
                    output1 = model.forward(data1, 0)
                    output2 = model.forward(data2, 1)

                    loss = F.binary_cross_entropy(torch.sigmoid(output1),target1) + reg*model.net.wConv1.norm() + reg*model.net.wConv2.norm() + reg*model.net1.wHidden.norm() + reg*model.net1.wNeu.norm()
                    loss += F.binary_cross_entropy(torch.sigmoid(output2),target2) + reg*model.net.wConv1.norm() + reg*model.net.wConv2.norm() + reg*model.net2.wHidden.norm() + reg*model.net2.wNeu.norm()
                    valid_loss.append(loss.cpu())

                    pred1_sig=torch.sigmoid(output1)
                    pred2_sig=torch.sigmoid(output2)

                    pred1 = pred1_sig.cpu().detach().numpy().reshape(output1.shape[0])
                    pred2 = pred2_sig.cpu().detach().numpy().reshape(output2.shape[0])

                    label1 = target1.cpu().numpy().reshape(output1.shape[0])
                    label2 = target2.cpu().numpy().reshape(output2.shape[0])

                    try:
                        auc1.append(metrics.roc_auc_score(label1, pred1))
                        auc2.append(metrics.roc_auc_score(label2, pred2))
                    except ValueError:
                        pass

                AUC_valid_1 = np.mean(auc1)
                AUC_valid_2 = np.mean(auc2)
                Loss_valid = np.mean(valid_loss)

            # training
            model.net.mode = 'training'
            model.net1.mode = 'trainig'
            model.net2.mode = 'training'
            for idx, (data, target, task) in enumerate(train_loader):

                data1, data2, target1, target2 = divide_batch(data, target, task, device)

                output1 = model.forward(data1, 0)
                output2 = model.forward(data2, 1)

                # task1 loss
                loss = F.binary_cross_entropy(torch.sigmoid(output1),target1) + reg*model.net.wConv1.norm() + reg*model.net.wConv2.norm() + reg*model.net1.wHidden.norm() + reg*model.net1.wNeu.norm()

                # task2 loss
                loss += F.binary_cross_entropy(torch.sigmoid(output2),target2) + reg*model.net.wConv1.norm() + reg*model.net.wConv2.norm() + reg*model.net2.wHidden.norm() + reg*model.net2.wNeu.norm()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            with open(result_path+'train/'+str(case_num)+'.txt', "a") as file:
                file.write('(')
                file.write(str(AUC_training_1))
                file.write(',')
                file.write(str(AUC_training_2))
                file.write(',')
                file.write(str(Loss_train))
                file.write(')')
                file.write('\n')
            file.close()
        
            with open(result_path+'valid/'+str(case_num)+'.txt', "a") as file:
                file.write('(')
                file.write(str(AUC_valid_1))
                file.write(',')
                file.write(str(AUC_valid_2))
                file.write(',')
                file.write(str(Loss_valid))
                file.write(')')
                file.write('\n')
            file.close()

            if Loss_valid < loss_best:
                loss_best = Loss_valid
                best_model = copy.deepcopy(model)
                state = {'conv1': model.net.wConv1,
                        'rect1':model.net.wRect1,
                        'conv2':model.net.wConv2,
                        'rect2':model.net.wRect2,
                        'wHidden1':model.net1.wHidden,
                        'wHiddenBias1':model.net1.wHiddenBias,
                        'wNeu1':model.net1.wNeu,
                        'wNeuBias1':model.net1.wNeuBias,
                        'wHidden2':model.net2.wHidden,
                        'wHiddenBias2':model.net2.wHiddenBias,
                        'wNeu2':model.net2.wNeu,
                        'wNeuBias2':model.net2.wNeuBias}

                isExist = os.path.exists('./MTL_Models/' + id)
                if not isExist:
                    os.makedirs('./MTL_Models/' + id)

                torch.save(state, './MTL_Models/' + id+ '/' + str(case_num+1) + '.pth')

        print('Training Completed')

        # Testing
        print('Model Testing')
        test_loader = test_dataset_loader_MT(tf1_test_dataset_path, tf2_test_dataset_path, motif_len)

        # using the model with best validation AUC
        model = best_model
        model.net.mode = 'test'
        model.net1.mode = 'test'
        model.net2.mode = 'test'

        with torch.no_grad():
            test_auc_1 = []
            test_auc_2 = []
            
            for idx, (data, target, task) in enumerate(test_loader):
                data1, data2, target1, target2 = divide_batch(data, target, task, device)

                # Forward pass
                output1 = model.forward(data1, 0)
                output2 = model.forward(data2, 1)

                pred1_sig=torch.sigmoid(output1)
                pred2_sig=torch.sigmoid(output2)

                pred1 = pred1_sig.cpu().detach().numpy().reshape(output1.shape[0])
                pred2 = pred2_sig.cpu().detach().numpy().reshape(output2.shape[0])

                label1 = target1.cpu().numpy().reshape(output1.shape[0])
                label2 = target2.cpu().numpy().reshape(output2.shape[0])

                try:
                    test_auc_1.append(metrics.roc_auc_score(label1, pred1))
                    test_auc_2.append(metrics.roc_auc_score(label2, pred2))
                except ValueError:
                    pass

            AUC_test_1 = np.mean(test_auc_1)
            AUC_test_2 = np.mean(test_auc_2)
            print('AUC on test data = ', AUC_test_1, AUC_test_2)

            with open(result_path+'test/'+str(case_num)+'.txt', "a") as file:
                file.write('AUC Test 1 : ')
                file.write(str(AUC_test_1))
                file.write(", ")
                file.write('AUC Test 2 : ')
                file.write(str(AUC_test_2))
                file.write('\n')
            file.close()

        print('Testing Completed')

def divide_batch(data, target, task, device):
    task1_data, task1_target = [], []
    task2_data, task2_target = [], []

    for i in range(len(task)):
        if task[i] == 0:
            task1_data.append(data[i].numpy())
            task1_target.append(target[i].numpy())
        elif task[i] == 1:
            task2_data.append(data[i].numpy())
            task2_target.append(target[i].numpy())
        else:
            print('?')
            quit()

    task1_data_array = np.array(task1_data)
    task2_data_array = np.array(task2_data)
    task1_data_tensor = torch.tensor(task1_data_array)
    task2_data_tensor = torch.tensor(task2_data_array)

    task1_target_array = np.array(task1_target)
    task2_target_array = np.array(task2_target)
    task1_target_tensor = torch.tensor(task1_target_array)
    task2_target_tensor = torch.tensor(task2_target_array)

    data1 = task1_data_tensor.to(device)
    data2 = task2_data_tensor.to(device)
    target1 = task1_target_tensor.to(device)
    target2 = task2_target_tensor.to(device)
    
    return data1, data2, target1, target2

if __name__ == '__main__':
    main()
