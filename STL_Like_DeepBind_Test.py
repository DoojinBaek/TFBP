import os
import time
import copy
import torch
import argparse
import datetime
import numpy as np
import torch.nn.functional as F

from sklearn import metrics
from TFBP import datasets, dataset_loader, test_dataset_loader, STL_Model, hyperparameters, write_settings, mkdir, write_train_valid_result
from TFBP import find_best_setting, write_settings_test, write_train_test_result, write_test_result

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
    name = tf[0]
    CodeTesting = args.codeTest
    id = args.id

    # Hyperparameters
    Num_Model = 6
    num_epochs = 100
    num_motif_detector = 16
    motif_len = 24
    batch_size = 64
    # reg = 1*10**-2
    reg = 2*10**-6

    best_hyperparameter_setting = find_best_setting(name, id, num_epochs)
    parsed = best_hyperparameter_setting.split(',')

    pool = parsed[0].split(' ')[-1]
    dropout_rate = float(parsed[1].split(' ')[-1])
    lr = float(parsed[2].split(' ')[-1])
    scheduler = bool(parsed[3].split(' ')[-1])
    opt = parsed[4].split(' ')[-1].split('\n')[0]

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
    _, _, _, _, _, _, train_loader = dataset_loader(train_dataset_path, batch_size)
    test_loader = test_dataset_loader(test_dataset_path, motif_len)

    print('Model Training with the Best Hyperparameter Setting')
    print(best_hyperparameter_setting.split('\n')[0])
    write_settings_test(name, id, pool, dropout_rate, lr, scheduler, opt)

    test_auc_best = 0

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
                test_auc = []
                test_loss_bce = []
                test_loss_rest = []

                for idx, (data, target) in enumerate(test_loader):
                    data = data.to(device)
                    target = target.to(device)

                    # Forward
                    output = model.forward(data)

                    test_loss_bce.append((F.binary_cross_entropy(torch.sigmoid(output),target)).cpu())
                    test_loss_rest.append((reg*model.base.wConv1.norm() + reg*model.base.wConv2.norm() + reg*model.fc.wHidden.norm() + reg*model.fc.wNeu.norm()).cpu())

                    pred_sig=torch.sigmoid(output)
                    pred=pred_sig.cpu().detach().numpy().reshape(output.shape[0])
                    labels=target.cpu().numpy().reshape(output.shape[0])
                    try:
                        test_auc.append(metrics.roc_auc_score(labels, pred))
                    except ValueError:
                        pass

                AUC_test = np.mean(test_auc)
                Loss_test_bce = np.mean(test_loss_bce)
                Loss_test_rest = np.mean(test_loss_rest)

                # write results
                write_train_test_result(name, id, AUC_training, Loss_training_bce, Loss_training_rest, AUC_test, Loss_test_bce, Loss_test_rest, model_num)

                # save model
                if test_auc_best < AUC_test:
                    test_auc_best = AUC_test
                    best_model = copy.deepcopy(model)
                    state = {'conv1': model.base.wConv1,
                            'rect1':model.base.wRect1,
                            'conv2': model.base.wConv2,
                            'rect2': model.base.wRect2,
                            'wHidden':model.fc.wHidden,
                            'wHiddenBias':model.fc.wHiddenBias,
                            'wNeu':model.fc.wNeu,
                            'wNeuBias':model.fc.wNeuBias}

                    if not os.path.exists('./Models/' + name + '/' + id):
                        os.makedirs('./Models/' + name + '/' + id)
                        
                    torch.save(state, './Models/' + name + '/' + id + '/' + 'best_model' + '.pth')

    print('Training Completed')

    # Report Performance

    print('Reporting Model Performance')

    model = best_model

    with torch.no_grad():
        model.base.mode = 'test'
        model.fc.mode = 'test'

        test_auc = []
        test_loss_bce = 0
        test_loss_rest = 0
        
        for idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)

            # Forward pass
            output = model.forward(data)

            test_loss_bce = (F.binary_cross_entropy(torch.sigmoid(output),target)).cpu()
            test_loss_rest = (reg*model.base.wConv1.norm() + reg*model.base.wConv2.norm() + reg*model.fc.wHidden.norm() + reg*model.fc.wNeu.norm()).cpu()
            
            pred_sig=torch.sigmoid(output)
            pred=pred_sig.cpu().detach().numpy().reshape(output.shape[0])
            labels=target.cpu().numpy().reshape(output.shape[0])
            try:
                test_auc.append(metrics.roc_auc_score(labels, pred))
            except ValueError:
                pass

        AUC_test=np.mean(test_auc)
        write_test_result(name, id, AUC_test, test_loss_bce, test_loss_rest)

if __name__ == '__main__':
    main()