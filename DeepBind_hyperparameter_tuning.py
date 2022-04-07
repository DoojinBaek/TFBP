import os
import time
import datetime

import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F

from sklearn import metrics
from utils import datasets, dataset_loader, test_dataset_loader
from network import  ConvNet_test

if(torch.cuda.is_available()):
    print('Torch',torch.__version__, 'is available')
else:
    print('Torch is not available. Process is terminated')
    quit()

def arg_parser():
    parser = argparse.ArgumentParser(description='Set hyperparameters or you can use default hyperparameter settings defined in the hyperparameter.json file')
    parser.add_argument('--TF', type=str, required=True, nargs=1, choices=['ARID3A', 'CTCFL', 'ELK1', 'FOXA1', 'GABPA', 'MYC', 'REST', 'SP1', 'USF1', 'ZBTB7A'], help='choose one from [ARID3A / CTCFL / ELK1 / FOXA1 / GABPA / MYC / REST / SP1 / USF1 / ZBTB7A]')
    parser.add_argument('--codeTest', type=bool, default=False, help='Do you want to test the code using only one hyperparameters setting?')
    parser.add_argument('--id', type=str, required=True, help='Set the name or id of this experiment')
    args = parser.parse_args()
    return args

def main():

    start = time.time()
    
    # parsing
    args = arg_parser()
    tf = args.TF
    CodeTesting = args.codeTest
    id = args.id
    print('TF Binding Prediction for', tf)
    print('Searching for all hyperparameter settings...')

    # Hyperparameters
    num_epochs = 150
    reverse_mode = False
    num_motif_detector = 16
    motif_len =24
    batch_size = 64
    beta1 = 2*10**-6
    beta2 = 5*10**-6
    beta3 = 2*10**-6
    if CodeTesting:
        pool_type = ['max']
        hidden_layer_type = [True] # add one hidden layer or not
        dropout_rate_type = [0.2]
        lr_type = [0.01]
        scheduler_type = [True] # use Cosine Annealing or not
        opt_type = ['Adam'] # optimizer
    else:
        pool_type = ['maxavg', 'max']
        hidden_layer_type = [True, False] # add one hidden layer or not
        dropout_rate_type = [0.2, 0.3, 0.4]
        lr_type_sgd = [0.001, 0.005, 0.01]
        lr_type_adam = [0.005, 0.01, 0.05]
        scheduler_type = [True, False] # use Cosine Annealing or not
        opt_type = ['SGD', 'Adam'] # optimizer

    total_cases = len(pool_type)*len(hidden_layer_type)*len(dropout_rate_type)*len(lr_type_sgd)*len(scheduler_type)*len(opt_type)

    print('Total cases :', total_cases)

    # Settings
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # dataset
    path = './data/encode/'
    all_dataset_names = datasets(path)
    TF2idx = {'ARID3A' : 0, 'CTCFL' : 1, 'ELK1' : 2, 'FOXA1' : 3, 'GABPA' : 4, 'MYC' : 5, 'REST' : 6, 'SP1' : 7, 'USF1' : 8, 'ZBTB7A' : 9}
    TFidx = TF2idx[tf[0]]
    dataset_name = all_dataset_names[TFidx]
    train_dataset_path = dataset_name[0]
    test_dataset_path = dataset_name[1]
    name = train_dataset_path.split(path)[1].split("_AC")[0]
    train_data_loader, valid_data_lodaer, all_train_data_lodaer = dataset_loader(train_dataset_path, batch_size, reverse_mode)

    for case_num in range(total_cases):
        print("---"*8)
        print(case_num+1, 'th experiment over ', total_cases)

        # specify hyperparameters
        (share, remainder) = divmod(case_num, len(opt_type))
        opt = opt_type[remainder]
        (share, remainder) = divmod(share, len(scheduler_type))
        scheduler = scheduler_type[remainder]
        (share, remainder) = divmod(share, len(lr_type_sgd))
        if(opt == 'SGD'):
            lr = lr_type_sgd[remainder]
        else:
            lr = lr_type_adam[remainder]
        (share, remainder) = divmod(share, len(dropout_rate_type))
        dropout_rate = dropout_rate_type[remainder]
        (share, remainder) = divmod(share, len(hidden_layer_type))
        hidden_layer = hidden_layer_type[remainder]
        (share, remainder) = divmod(share, len(pool_type))
        pool = pool_type[remainder]

        with open("./results/epochs_experiments/"+name+'-'+id+'.txt', "a") as file:
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
            file.write(', hidden layer : ')
            file.write(str(hidden_layer))
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

        model = ConvNet_test(num_motif_detector,motif_len,pool,hidden_layer,'training',lr,opt, dropout_rate,beta1,beta2,beta3,device,reverse_complemet_mode=False).to(device)

        # optimizer
        if hidden_layer == True:
            if opt == 'SGD':
                optimizer = torch.optim.SGD([model.wConv,model.wRect,model.wNeu,model.wNeuBias,model.wHidden,model.wHiddenBias] , lr = lr, momentum = 0.9) 
            else:
                optimizer = torch.optim.SGD([model.wConv,model.wRect,model.wNeu,model.wNeuBias,model.wHidden,model.wHiddenBias], lr = lr)
        else:
            if opt == 'SGD':
                optimizer = torch.optim.SGD([model.wConv,model.wRect,model.wNeu,model.wNeuBias] , lr = lr, momentum = 0.9) 
            else:
                optimizer = torch.optim.SGD([model.wConv,model.wRect,model.wNeu,model.wNeuBias], lr = lr)

        # scheduler
        if scheduler == True:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
        else:
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1) # constant learning rate

        train_loader = train_data_loader
        valid_loader = valid_data_lodaer

        best_AUC = 0
        best_epoch = 0

        with open("./results/epochs_experiments/"+name+'-'+id+'.txt', "a") as file:
            file.write('Train AUC over epochs : ')
        file.close()

        for epoch in range(num_epochs):
            if epoch%10 == 0:
                print(epoch, 'th epoch over ', num_epochs)
            for idx, (data, target) in enumerate(train_loader):
                # idx -> ceiling(#data/batch)
                data = data.to(device)
                target = target.to(device)
                if model.reverse_complemet_mode:
                    target_2=torch.randn(int(target.shape[0]/2),1)
                    for i in range(target_2.shape[0]):
                        target_2[i]=target[2*i]
                    target=target_2.to(device)

                # Forward pass
                output = model(data)
                
                if hidden_layer == True:
                    loss = F.binary_cross_entropy(torch.sigmoid(output),target)+model.beta1*model.wConv.norm()+model.beta2*model.wHidden.norm()+model.beta3*model.wNeu.norm()
                else:
                    loss = F.binary_cross_entropy(torch.sigmoid(output),target)+model.beta1*model.wConv.norm()+model.beta3*model.wNeu.norm()
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
                
            with torch.no_grad():
                # for train set
                model.mode='test'
                auc=[]
                for idx, (data, target) in enumerate(train_loader):
                    data = data.to(device)
                    target = target.to(device)
                    if model.reverse_complemet_mode:
                        target_2=torch.randn(int(target.shape[0]/2), 1)
                        for i in range(target_2.shape[0]):
                            target_2[i]=target[2*i]
                        target=target_2.to(device)

                    # Forward pass
                    output = model(data)
                    pred_sig=torch.sigmoid(output)
                    pred=pred_sig.cpu().detach().numpy().reshape(output.shape[0])
                    labels=target.cpu().numpy().reshape(output.shape[0])
                    try:
                        auc.append(metrics.roc_auc_score(labels, pred))
                    except ValueError:
                        pass

                AUC_training=np.mean(auc)

                # for valid set
                model.mode='test'
                auc=[]
                for idx, (data, target) in enumerate(valid_loader):
                    data = data.to(device)
                    target = target.to(device)
                    if model.reverse_complemet_mode:
                        target_2=torch.randn(int(target.shape[0]/2), 1)
                        for i in range(target_2.shape[0]):
                            target_2[i]=target[2*i]
                        target=target_2.to(device)

                    # Forward pass
                    output = model(data)
                    pred_sig=torch.sigmoid(output)
                    pred=pred_sig.cpu().detach().numpy().reshape(output.shape[0])
                    labels=target.cpu().numpy().reshape(output.shape[0])
                    try:
                        auc.append(metrics.roc_auc_score(labels, pred))
                    except ValueError:
                        pass

                AUC_valid=np.mean(auc)

                with open("./results/epochs_experiments/"+name+'-'+id+'.txt', "a") as file:
                    file.write(str(AUC_training))
                    file.write(':')
                    file.write(str(AUC_valid))
                    file.write(', ')
                file.close()

                if AUC_valid > best_AUC:
                    best_AUC = AUC_valid
                    best_epoch = epoch
                    best_model = model
                    state = {'conv': model.wConv,
                            'rect':model.wRect,
                            'wHidden':model.wHidden,
                            'wHiddenBias':model.wHiddenBias,
                            'wNeu':model.wNeu,
                            'wNeuBias':model.wNeuBias}

                    isExist = os.path.exists('./Models/' + name + '/' + id)
                    if not isExist:
                        os.makedirs('./Models/' + name + '/' + id)

                    torch.save(state, './Models/' + name + '/' + id + '/' + str(case_num+1) + '.pth')
        
        with open("./results/epochs_experiments/"+name+'-'+id+'.txt', "a") as file:
            file.write('\n')
            file.write('Best Trainig(validation) AUC : ')
            file.write(str(best_AUC))
            file.write(' at ')
            file.write(str(best_epoch))
            file.write('th epoch')
            file.write('\n')
        file.close()

        print('Training Completed')
        
        # Testing

        print('Model Testing')

        test_loader = test_dataset_loader(test_dataset_path, motif_len)

        # using the model with best validation AUC
        model = best_model

        with torch.no_grad():
            model.mode='test'
            auc=[]
            
            for idx, (data, target) in enumerate(test_loader):
                data = data.to(device)
                target = target.to(device)
                if model.reverse_complemet_mode:
                    target_2=torch.randn(int(target.shape[0]/2),1)
                    for i in range(target_2.shape[0]):
                        target_2[i]=target[2*i]
                    target=target_2.to(device)

                # Forward pass
                output = model(data)
                pred_sig=torch.sigmoid(output)
                pred=pred_sig.cpu().detach().numpy().reshape(output.shape[0])
                labels=target.cpu().numpy().reshape(output.shape[0])
                try:
                    auc.append(metrics.roc_auc_score(labels, pred))
                except ValueError:
                    pass

            AUC_test=np.mean(auc)
            # print('AUC on test data = ', AUC_test)
            with open("./results/epochs_experiments/"+name+'-'+id+'.txt', "a") as file:
                file.write('AUC Test : ')
                file.write(str(round(AUC_test, 5)))
                file.write('\t')
                file.write('Elapsed Time : ')
                file.write(str(datetime.timedelta(seconds = (time.time()-start))))
                file.write('\n')
            file.close()
    
    print('Testing Completed')

if __name__ == '__main__':
    main()
