import os
import time
import torch
import argparse
import datetime
import numpy as np
import torch.nn.functional as F

from sklearn import metrics
from TFBP import datasets, dataset_loader, test_dataset_loader, STL_Model

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
    num_motif_detector = 16
    motif_len = 24
    batch_size = 64
    reg = 2*10**-6

    if CodeTesting:
        pool_type = ['max']
        dropout_rate_type = [0.2]
        lr_type_adam = [0.01]
        scheduler_type = [True] # use Cosine Annealing or not
        opt_type = ['Adam'] # optimizer
    else:
        pool_type = ['maxavg', 'max']
        dropout_rate_type = [0.2, 0.3, 0.4]
        lr_type_sgd = [0.001, 0.005, 0.01]
        lr_type_adam = [0.005, 0.01, 0.05]
        scheduler_type = [True, False] # use Cosine Annealing or not
        opt_type = ['SGD', 'Adam'] # optimizer
    
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
    test_dataset_path = dataset_name[1]
    name = tf[0]
    train_data_loader, valid_data_lodaer, all_train_data_lodaer = dataset_loader(train_dataset_path, batch_size, reverse_mode=False)

    for case_num in range(total_cases):
        print("---"*8)
        print(case_num+1, 'th experiment over ', total_cases)

        # specify hyperparameters
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

        if not os.path.exists("./results/"+name+'-'+id+'/'+'experiemt/'):
            os.makedirs("./results/"+name+'-'+id+'/'+'experiemt/')
        if not os.path.exists("./results/"+name+'-'+id+'/'+'train/'):
            os.makedirs("./results/"+name+'-'+id+'/'+'train/')
        if not os.path.exists("./results/"+name+'-'+id+'/'+'valid/'):
            os.makedirs("./results/"+name+'-'+id+'/'+'valid/')
        if not os.path.exists("./results/"+name+'-'+id+'/'+'test/'):
            os.makedirs("./results/"+name+'-'+id+'/'+'test/')

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

        # Model Training

        print('Model Training')

        model = STL_Model(num_motif_detector,motif_len,pool,'training',lr, dropout_rate,device)

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

        train_loader = train_data_loader
        valid_loader = valid_data_lodaer

        Loss_best = 1000 # big enough

        for epoch in range(num_epochs):
            
            model.base.mode = 'training'
            model.fc.mode = 'training'

            if epoch%10 == 0:
                print(epoch, 'th epoch over ', num_epochs)
            for idx, (data, target) in enumerate(train_loader):
                # idx -> ceiling(#data/batch)
                data = data.to(device)
                target = target.to(device)

                # Forward pass
                output = model.forward(data)

                loss = F.binary_cross_entropy(torch.sigmoid(output),target) + reg*model.base.wConv1.norm() + reg*model.base.wConv2.norm() + reg*model.fc.wHidden.norm() + reg*model.fc.wNeu.norm()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
                
            with torch.no_grad():
                # for train set
                model.base.mode = 'test'
                model.fc.mode = 'test'

                train_auc = []
                train_loss_bce = []
                train_loss_rest = []
                for idx, (data, target) in enumerate(train_loader):
                    data = data.to(device)
                    target = target.to(device)

                    # Forward pass
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
                Loss_trainin_rest = np.mean(train_loss_rest)

                # for valid set
                model.base.mode = 'test'
                model.fc.mode = 'test'
                
                valid_auc = []
                valid_loss = []
                for idx, (data, target) in enumerate(valid_loader):
                    data = data.to(device)
                    target = target.to(device)

                    # Forward pass
                    output = model.forward(data)
                    valid_loss.append((F.binary_cross_entropy(torch.sigmoid(output),target) + reg*model.base.wConv1.norm() + reg*model.base.wConv2.norm() + reg*model.fc.wHidden.norm() + reg*model.fc.wNeu.norm()).cpu())
                    pred_sig=torch.sigmoid(output)
                    pred=pred_sig.cpu().detach().numpy().reshape(output.shape[0])
                    labels=target.cpu().numpy().reshape(output.shape[0])
                    try:
                        valid_auc.append(metrics.roc_auc_score(labels, pred))
                    except ValueError:
                        pass

                AUC_valid = np.mean(valid_auc)
                Loss_valid = np.mean(valid_loss)

                with open("./results/"+name+'-'+id+'/'+'train/'+str(case_num)+'.txt', "a") as file:
                    file.write(str(AUC_training))
                    file.write(':')
                    file.write(str(Loss_training_bce))
                    file.write('+')
                    file.write(str(Loss_trainin_rest))
                    file.write('\n')
                file.close()
                with open("./results/"+name+'-'+id+'/'+'valid/'+str(case_num)+'.txt', "a") as file:
                    file.write(str(AUC_valid))
                    file.write(':')
                    file.write(str(Loss_valid))
                    file.write('\n')
                file.close()

                if Loss_valid < Loss_best: # update the model based on the loss
                    Loss_best = Loss_valid
                    best_model = model
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
                    torch.save(state, './Models/' + name + '/' + id + '/' + str(case_num+1) + '.pth')

        print('Training Completed')
        
        # Testing

        print('Model Testing')

        test_loader = test_dataset_loader(test_dataset_path, motif_len)

        # using the model with best validation AUC
        model = best_model

        with torch.no_grad():
            model.base.mode = 'test'
            model.fc.mode = 'test'

            test_auc = []
            test_loss = 0
            
            for idx, (data, target) in enumerate(test_loader):
                data = data.to(device)
                target = target.to(device)

                # Forward pass
                output = model.forward(data)
                test_loss = (F.binary_cross_entropy(torch.sigmoid(output),target) + reg*model.base.wConv1.norm() + reg*model.base.wConv2.norm() + reg*model.fc.wHidden.norm() + reg*model.fc.wNeu.norm()).cpu()
                pred_sig=torch.sigmoid(output)
                pred=pred_sig.cpu().detach().numpy().reshape(output.shape[0])
                labels=target.cpu().numpy().reshape(output.shape[0])
                try:
                    test_auc.append(metrics.roc_auc_score(labels, pred))
                except ValueError:
                    pass

            AUC_test=np.mean(test_auc)

            with open("./results/"+name+'-'+id+'/'+'test/'+str(case_num)+'.txt', "a") as file:
                file.write('Test AUC : ')
                file.write(str(AUC_test))
                file.write('\n')
                file.write('Test Loss :')
                file.write(str(test_loss.item()))
                file.write('\n')
                file.write('Elapsed Time : ')
                file.write(str(datetime.timedelta(seconds = (time.time()-start))))
                file.write('\n')
            file.close()
    
    print('Testing Completed')

if __name__ == '__main__':
    main()
