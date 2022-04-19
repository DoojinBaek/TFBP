import os
import time
import torch
import argparse
import datetime
import numpy as np
import torch.nn.functional as F

from sklearn import metrics
from TFBP import datasets, dataset_loader, STL_Model, hyperparameters, write_settings, mkdir, write_train_valid_result

if(torch.cuda.is_available()):
    print('Torch',torch.__version__, 'is available')
else:
    print('Torch is not available. Process is terminated')
    quit()

def arg_parser():
    parser = argparse.ArgumentParser(description='Set hyperparameters or you can use default hyperparameter settings defined in the hyperparameter.json file')
    parser.add_argument('--TF', type=str, required=True, nargs=1, choices=['ARID3A', 'CTCFL', 'ELK1', 'FOXA1', 'GABPA', 'MYC', 'REST', 'SP1', 'USF1', 'ZBTB7A'], help='choose one from [ARID3A / CTCFL / ELK1 / FOXA1 / GABPA / MYC / REST / SP1 / USF1 / ZBTB7A]')
    parser.add_argument('--id', type=str, required=True, help='Set the name or id of this experiment')
    parser.add_argument('--reg', type=int, required=True)
    args = parser.parse_args()
    return args

def main():

    start = time.time()
    
    # parsing
    args = arg_parser()
    tf = args.TF
    id = args.id
    lambda_input = args.reg
    print('TF Binding Prediction for', tf)
    print('Searching for all hyperparameter settings...')

    # Hyperparameters
    Num_Model = 3
    num_epochs = 150
    num_motif_detector = 16
    motif_len = 24
    batch_size = 64
    # reg = 1*10**-2
    # reg = 2*10**-6
    if lambda_input == 1:
        reg = 4*10**-2
    elif lambda_input == 10:
        reg = 4*10**-3
    elif lambda_input == 100:
        reg = 4*10**-4
    elif lambda_input == 1000:
        reg = 4*10**-5
    elif lambda_input == 10000:
        reg = 4*10**-6
    elif lambda_input == 20000:
        reg = 2*10**-6

    # pool_type = ['max']
    # dropout_rate_type = [1.0] # 1.0 for no-dropout
    # lr_type_sgd = [0.001, 0.005, 0.01, 0.05]
    # lr_type_adam = [0.05]
    # scheduler_type = [False] # use Cosine Annealing or not
    # opt_type = ['Adam'] # optimizer

    # MYC Best Settings
    pool_type = ['maxavg']
    dropout_rate_type = [0.5] # 1.0 for no-dropout
    lr_type_sgd = [0.01]
    lr_type_adam = [0.05]
    scheduler_type = [True] # use Cosine Annealing or not
    opt_type = ['SGd'] # optimizer

    total_cases = len(pool_type)*len(dropout_rate_type)*len(lr_type_adam)*len(scheduler_type)*len(opt_type)
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
    name = tf[0]
    train_loader_1, valid_loader_1, train_loader_2, valid_loader_2, train_loader_3, valid_loader_3, _ = dataset_loader(train_dataset_path, batch_size)
    train_loader_set = [train_loader_1, train_loader_2, train_loader_3]
    valid_loader_set = [valid_loader_1, valid_loader_2, valid_loader_3]

    for case_num in range(total_cases):
        print("---"*8)
        print(case_num+1, 'th experiment over ', total_cases)

        # specify hyperparameters
        opt, scheduler, lr, dropout_rate, pool = hyperparameters(case_num, False, opt_type, scheduler_type, lr_type_adam, lr_type_sgd, dropout_rate_type, pool_type)

        mkdir(name, id)
        write_settings(name, id, case_num, total_cases, pool, dropout_rate, lr, scheduler, opt)

        for model_num in range(Num_Model):
            # Model Training
            print('Training Model', model_num+1)

            model = STL_Model(num_motif_detector, motif_len, pool, 'training', lr, dropout_rate, device)

            # optimizer
            if opt == 'SGD':
                optimizer = torch.optim.SGD([model.base.wConv1, model.base.wRect1, model.base.wConv2, model.base.wRect2, model.fc.wNeu, model.fc.wNeuBias, model.fc.wHidden, model.fc.wHiddenBias], lr = lr, momentum = 0.9) 
            else:
                optimizer = torch.optim.SGD([model.base.wConv1, model.base.wRect1, model.base.wConv2, model.base.wRect2, model.fc.wNeu, model.fc.wNeuBias, model.fc.wHidden, model.fc.wHiddenBias], lr = lr)

            # scheduler
            if scheduler == True:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
            else:
                scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1) # constant learning rate
            
            train_loader = train_loader_set[model_num]
            valid_loader = valid_loader_set[model_num]

            for epoch in range(num_epochs):
                if epoch%10 == 0:
                    print(epoch, 'th epoch over', num_epochs)

                # training
                for idx, (data, target) in enumerate(train_loader):

                    model.base.mode = 'training'
                    model.fc.mode = 'training'

                    data = data.to(device)
                    target = target.to(device)
                    
                    # Forward
                    output = model.forward(data)
                    loss = F.binary_cross_entropy(torch.sigmoid(output),target) + reg*model.base.wConv1.norm() + reg*model.base.wConv2.norm() + reg*model.fc.wHidden.norm() + reg*model.fc.wNeu.norm()

                    # Backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                
                with torch.no_grad():
                # save model performance
                    model.base.mode = 'test'
                    model.fc.mode = 'test'

                    # For training set
                    train_auc = []
                    train_loss_bce = []
                    train_loss_rest = []

                    for idx, (data, target) in enumerate(train_loader):
                        data = data.to(device)
                        target = target.to(device)

                        # Forward
                        output = model.forward(data)

                        train_loss_bce.append((F.binary_cross_entropy(torch.sigmoid(output),target)).cpu())
                        train_loss_rest.append((reg*model.base.wConv1.norm() + reg*model.base.wConv2.norm() + reg*model.fc.wHidden.norm() + reg*model.fc.wNeu.norm()).cpu())

                        pred_sig=torch.sigmoid(output)
                        pred=pred_sig.cpu().detach().numpy().reshape(output.shape[0])
                        labels=target.cpu().numpy().reshape(output.shape[0])
                        try:
                            train_auc.append(metrics.roc_auc_score(labels, pred))
                        except ValueError:
                            pass

                    AUC_training = np.mean(train_auc)
                    Loss_training_bce = np.mean(train_loss_bce)
                    Loss_training_rest = np.mean(train_loss_rest)

                    # For validation set
                    valid_auc = []
                    valid_loss_bce = []
                    valid_loss_rest = []

                    for idx, (data, target) in enumerate(valid_loader):
                        data = data.to(device)
                        target = target.to(device)

                        # Forward
                        output = model.forward(data)

                        valid_loss_bce.append((F.binary_cross_entropy(torch.sigmoid(output),target)).cpu())
                        valid_loss_rest.append((reg*model.base.wConv1.norm() + reg*model.base.wConv2.norm() + reg*model.fc.wHidden.norm() + reg*model.fc.wNeu.norm()).cpu())

                        pred_sig=torch.sigmoid(output)
                        pred=pred_sig.cpu().detach().numpy().reshape(output.shape[0])
                        labels=target.cpu().numpy().reshape(output.shape[0])
                        try:
                            valid_auc.append(metrics.roc_auc_score(labels, pred))
                        except ValueError:
                            pass

                    AUC_validation = np.mean(valid_auc)
                    Loss_validation_bce = np.mean(valid_loss_bce)
                    Loss_validation_rest = np.mean(valid_loss_rest)

                    # write results
                    write_train_valid_result(name, id, case_num, AUC_training, Loss_training_bce, Loss_training_rest, AUC_validation, Loss_validation_bce, Loss_validation_rest, model_num, epoch)

        print('Training Completed')

if __name__ == '__main__':
    main()
    exec(open('STL_Like_DeepBind_Test_simple.py').read())
