import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, average_precision_score
import pickle
import time
import shutil

from config import Config
from utils_func import *
from utils_data import DLoader
from model import ResNet



class Trainer:
    def __init__(self, config:Config, device:torch.device, mode:str, continuous:int):
        self.config = config
        self.device = device
        self.mode = mode
        self.continuous = continuous
        self.dataloaders = {}

        # if continuous, load previous training info
        if self.continuous:
            with open(self.config.loss_data_path, 'rb') as f:
                self.loss_data = pickle.load(f)

        # path, data params
        self.base_path = self.config.base_path
        self.model_path = self.config.model_path

        # train params
        self.batch_size = self.config.batch_size
        self.epochs = self.config.epochs
        self.lr = self.config.lr

        # data condition
        self.img_size = self.config.img_size
        self.label = self.config.label

        # make data frame
        meta_path = self.base_path + '/data/mimic-cxr-2.0.0-metadata.csv'
        negbio_path = self.base_path + '/data/mimic-cxr-2.0.0-negbio.csv'
        meta, negbio = make_dataframe(meta_path, negbio_path)

        # extract necessary image data
        img_dict = make_img_data(meta)
        label_dict = make_label(negbio)

        # merge dataset
        dataset = merge_data(img_dict, label_dict)

        if not os.path.isdir(self.base_path + '/data/train'):
            # download the chest X-ray images
            if not os.path.isdir(self.base_path + '/physionet.org'):
                download_img(self.base_path, dataset, self.config.physio_id, self.config.pwd)
                msg = """
                        please copy and paste the command in {base_path}/cmd.txt to your terminal.
                        It will take a lot of time to download the images.
                        """
                print(msg)
                sys.exit()
            # split to train and test
            else:
                split_data(self.base_path, dataset)
                shutil.rmtree(self.base_path + '/physionet.org/')
                os.remove(self.base_path + '/cmd.txt')
    
        # sanity check
        assert len(dataset) == len(os.listdir(self.base_path + '/data/train')) + len(os.listdir(self.base_path + '/data/test'))
        coef = get_coef(dataset).to(self.device)

        # image preprocessing
        m, std = 0.4690693035517775, 0.3045583482083454
        self.trans = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([m], [std])
            ])

        if self.mode == 'train':
            self.trainset = DLoader(self.base_path, dataset, 'train', self.trans)
            self.testset = DLoader(self.base_path, dataset, 'test', self.trans)
            self.dataloaders['train'] = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=4)
            self.dataloaders['test'] = DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        else:
            self.testset = DLoader(self.base_path, dataset, 'test', self.trans)
            self.dataloaders['test'] = DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        
        # model, optimizer, loss
        self.model = ResNet(self.img_size, self.label).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=coef) if self.config.pos_wts else nn.BCEWithLogitsLoss()
        if self.mode == 'train':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            if self.continuous:
                self.check_point = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(self.check_point['model'])
                self.optimizer.load_state_dict(self.check_point['optimizer'])
                del self.check_point
                torch.cuda.empty_cache()
        elif self.mode == 'test':
            self.check_point = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(self.check_point['model'])
            self.model.eval()
            del self.check_point
            torch.cuda.empty_cache()

        
    def train(self):
        early_stop = 0
        best_acc = 0 if not self.continuous else self.loss_data['best_acc']
        train_loss_history = [] if not self.continuous else self.loss_data['train_loss_history']
        val_loss_history = [] if not self.continuous else self.loss_data['val_loss_history']
        best_epoch_info = 0 if not self.continuous else self.loss_data['best_epoch']

        for epoch in range(self.epochs):
            start = time.time()
            print(epoch+1, '/', self.epochs)
            print('-'*10)
            for phase in ['train', 'test']:
                print('Phase: {}'.format(phase))
                if phase == 'train':
                    self.model.train()                
                else:
                    self.model.eval()                    

                total_loss, total_acc = 0, 0
                for i, (img, label) in enumerate(self.dataloaders[phase]):
                    batch_size = img.size(0)
                    img, label = img.to(self.device), label.to(self.device)
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase=='train'):
                        output = self.model(img)
                        loss = self.criterion(output, label)
                        predict = (F.sigmoid(output) > 0.5).float()
                        acc = (predict==label).float().sum()/batch_size/self.label
                        
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                        
                    total_loss += loss.item()*batch_size
                    total_acc += acc * batch_size
                    if i % 100 == 0:
                        print('Epoch {}: {}/{} step loss: {}, acc: {}'.format(epoch+1, i, len(self.dataloaders[phase]), loss.item(), acc))
                epoch_loss = total_loss/len(self.dataloaders[phase].dataset)
                epoch_acc = total_acc/len(self.dataloaders[phase].dataset)
                # one_acc = pred1/total1
                print('{} loss: {:4f}, acc: {:4f}\n'.format(phase, epoch_loss, epoch_acc))

                if phase == 'train':
                    train_loss_history.append(epoch_loss)
                if phase == 'test':
                    val_loss_history.append(epoch_loss)

                    # save best model
                    early_stop += 1
                    if epoch_acc > best_acc:
                        early_stop = 0
                        best_acc = epoch_acc
                        best_epoch = best_epoch_info + epoch + 1
                        save_checkpoint(self.model_path, self.model, self.optimizer)

            print("time: {} s\n".format(time.time() - start))
            print('\n'*2)

            # early stopping
            if early_stop == self.config.early_stop_criterion:
                break

        print('best val acc: {:4f}, best epoch: {:d}\n'.format(best_acc, best_epoch))
        self.loss_data = {'best_epoch': best_epoch, 'best_acc': best_acc, 'train_loss_history': train_loss_history, 'val_loss_history': val_loss_history}
        return self.loss_data
    

    def test(self):        
        # statistics of the test set
        phase = 'test'
        total_y, total_output = [], []
        with torch.no_grad():
            self.model.eval()
            total_loss, total_acc = 0, 0
            for img, label in self.dataloaders[phase]:
                batch_size = img.size(0)
                img, label = img.to(self.device), label.to(self.device)
                
                output = self.model(img)
                loss = self.criterion(output, label)

                total_loss += loss.item()*batch_size
                predict = (F.sigmoid(output) > 0.5).float()
                total_acc += (predict==label).float().sum().item()/self.label

                total_y.append(label.detach().cpu())
                total_output.append(F.sigmoid(output).detach().cpu())

            total_y = torch.cat(total_y, dim=0).numpy()
            total_output = torch.cat(total_output, dim=0).numpy()

            total_loss = total_loss / len(self.dataloaders[phase].dataset)
            total_acc = total_acc/len(self.dataloaders[phase].dataset)
            print('test loss: {:4f}, test acc: {}'.format(total_loss, total_acc))

            print('AUROC of macro test data: ', roc_auc_score(total_y, total_output, average='macro'))
            print('AUROC of micro test data: ', roc_auc_score(total_y, total_output, average='micro'))
            print('AUPRC of macro test data: ', average_precision_score(total_y, total_output, average='macro'))
            print('AUPRC of micro test data: ', average_precision_score(total_y, total_output, average='micro'))