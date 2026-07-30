from forecasting.data_provider.data_factory import data_provider
from forecasting.exp.exp_basic import Exp_Basic
from forecasting.models import DLinear, Linear, NLinear, PatchTST, TSMixer, TimeMixer, iTransformers
from models import Autoformer
from forecasting.utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from forecasting.utils.metrics import metric
from sklearn.metrics import mean_squared_error, mean_absolute_error


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pdb

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'TSMixer' : TSMixer,
            'PatchTST' : PatchTST,
            'iTransformers': iTransformers,
            'TimeMixer': TimeMixer,
            'Autoformer': Autoformer
        }

        self.non_transformer_model_list = {'DLinear', 'NLinear', 'Linear', 'TSMixer', 'Crossformer', 'PatchTST', 'Autoformer'} 
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data = None, vali_loader= None, criterion = None, vali_iterator = None):
        total_loss = []
        total_loss_mae = []
        self.model.eval()
        vali_steps = self.args.seq_len
        iter_count = 0
        criterion_mae = nn.L1Loss()
 
        with torch.no_grad():
            
           
            if 'Cycle' not in self.args.model:
                for i, data in enumerate(vali_loader):
                    
                    
                    batch_x, batch_y, batch_x_mark, batch_y_mark = data
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float()

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    

                    if (self.args.model == "WaveForM") and (batch_x.size(0) < self.args.batch_size) :
                        break
                    
                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.model in self.non_transformer_model_list :
                                
                                
                                outputs = self.model(batch_x)
                                
                            else:
                                if self.args.output_attention and self.args.model == 'iTransformer':
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                    
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.model in self.non_transformer_model_list :
                        
                            
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention and self.args.model == "iTransformers":
                                outputs, attention_weight = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                
                    
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()
                    
                    loss_mae = criterion_mae(pred, true)
                    loss = criterion(pred, true)
                
                    total_loss_mae.append(loss_mae)
                    total_loss.append(loss)
                
                
            else:
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(vali_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float()

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    batch_cycle = batch_cycle.int().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if any(substr in self.args.model for substr in {'Cycle'}):
                                outputs = self.model(batch_x, batch_cycle)
                            elif any(substr in self.args.model for substr in
                                    {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF'}):
                                outputs = self.model(batch_x)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if any(substr in self.args.model for substr in {'Cycle'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()
                    
                    loss_mae = criterion_mae(pred, true)
                    loss = criterion(pred, true)
                
                    total_loss_mae.append(loss_mae)
                    total_loss.append(loss)
                
                    
           
        if (len(total_loss_mae) > 0) :
            total_loss_mae = np.average(total_loss_mae)
        else :
            total_loss_mae = 0
                    
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss, total_loss_mae

    def train(self, setting):
        
        train_data, train_loader = self._get_data(flag='train')
        
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')
        
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            print(path)
            os.makedirs(path,  exist_ok=True)
            
        if not hasattr(self.args, "trial_id"):
            self.args.trial_id = 999
            
        self.args.ckpt_path = os.path.join(path, f"trial_{self.args.trial_id}_checkpoint.pth")
        
        

        time_now = time.time()

        
        train_steps = len(train_loader)
            
            
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
       
       
        criterion = self._select_criterion()
            
            
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()  

        # get model  parameter count
        pytorch_total_params_size = sum((p.numel() * p.element_size())/ 1024**2 for p in self.model.parameters())
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        print(f"model has {pytorch_total_params_size} MB size and {pytorch_total_params} params")
        print(f"Initial GPU memory usage: {torch.cuda.memory_allocated() / 2**30} GB")

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            
            if 'Cycle' not in self.args.model:
 
                for i, data in enumerate(train_loader):
                    iter_count += 1
                    model_optim.zero_grad()
             
                    batch_x, batch_y, batch_x_mark, batch_y_mark = data
                    batch_x = batch_x.float().to(self.device)
            

                    if (self.args.model == "WaveForM") and (batch_x.size(0) < self.args.batch_size):
                        break

                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            
                            if self.args.model in self.non_transformer_model_list:

                                outputs = self.model(batch_x)
                            
                            else:
                                if self.args.output_attention and (self.args.model == "iTransformers"):

                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                    
                    
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            
                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss = criterion(outputs, batch_y)
                            train_loss.append(loss.item())
                                    
                           
                    else:
                        if self.args.model in self.non_transformer_model_list:  
                          
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention and (self.args.model == "iTransformers"):

                                outputs, attention_weight = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    
       
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    
                        
                        f_dim = -1 if self.args.features == 'MS' else 0
                        
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
  
                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()
                        
            else:
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                    iter_count += 1
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)

                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    batch_cycle = batch_cycle.int().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if any(substr in self.args.model for substr in {'Cycle'}):
                                outputs = self.model(batch_x, batch_cycle)
                            elif any(substr in self.args.model for substr in
                                    {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF'}):
                                outputs = self.model(batch_x)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss = criterion(outputs, batch_y)
                            train_loss.append(loss.item())
                    else:
                        if any(substr in self.args.model for substr in {'Cycle'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())

                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()
                    

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            if not self.args.train_only:
                
         
                vali_loss, _ = self.vali(vali_data=vali_data, vali_loader=vali_loader, criterion=criterion)
                test_loss, test_loss_mae = self.vali(vali_data=test_data, vali_loader=test_loader, criterion=criterion)
                
                    

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f} Test MAE: {5:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss, test_loss_mae))

                early_stopping(vali_loss, self.model, self.args.ckpt_path)
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))

                early_stopping(vali_loss, self.model, self.args.ckpt_path)

            if early_stopping.early_stop and self.args.early_stopping:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
        
        vali_loss, vali_loss_mae = self.vali(vali_data=vali_data, vali_loader=vali_loader, criterion=criterion)
            
        vali_loss = (vali_loss, vali_loss_mae)


        return self.model, vali_loss


    def test(self, setting, test=0, save_visualize=False, model=None):
 
        test_data, test_loader = self._get_data(flag='test')
            
        window_sampling_limit = self.args.seq_len * self.args.pred_len
            
        if test:
            print('loading model')
            
            checkpoint_path = os.path.join(self.args.checkpoints, setting, f"trial_{self.args.trial_id}_checkpoint.pth")

            if not os.path.exists(checkpoint_path):
     
                generic_checkpoint_path = os.path.join(checkpoint_path, 'checkpoint.pth')
                if os.path.exists(generic_checkpoint_path):
                    print(f"Warning: Trial-specific checkpoint not found. Falling back to generic '{generic_checkpoint_path}'")
                    checkpoint_path = generic_checkpoint_path
                else:
                    print(f"Error: Checkpoint file not found at '{checkpoint_path}'")
                    return 
            
            print(f"Loading checkpoint from: {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))


        preds = []
        trues = []
        inputx = []
    
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        attention_weights = []
        iter_count = 0
        with torch.no_grad():
            
            
            if 'Cycle' not in self.args.model:
 
                for i, data in enumerate(test_loader):

                    batch_x, batch_y, batch_x_mark, batch_y_mark = data
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    
                    if (self.args.model == "WaveForM") and (batch_x.size(0) < self.args.batch_size) :
                        break
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    
                    if (self.args.model == "WaveForM") and (batch_x.size(0) < self.args.batch_size) :
                        break

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.model in self.non_transformer_model_list:
                                
                                outputs = self.model(batch_x)
                            else:
                                if self.args.output_attention and  self.args.model == "iTransformers":
                                    outputs, attention_weight = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                    
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.model in self.non_transformer_model_list:
                        
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention and self.args.model == "iTransformers":
                                outputs, attention_weight = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            

                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()
                

                    pred = outputs
                    true = batch_y
        

                    if self.args.model == "iTransformers" :
                        attention_weight = attention_weight.detach().cpu()
                        attention_weights.append(attention_weight)
                    preds.append(pred)
                    trues.append(true)
                    inputx.append(batch_x.detach().cpu().numpy())
                    if i % 20 == 0 and save_visualize:
                        input = batch_x.detach().cpu().numpy()
                        gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                            
                    elif self.args.model == 'Nhits':
                        outsample_y, forecast, outsample_mask = outputs
                        pred = forecast.detach().cpu().numpy()
                        true = outsample_y.detach().cpu().numpy()
                        preds.append(pred)
                        trues.append(true)
                        
                        
            else:
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    batch_cycle = batch_cycle.int().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if any(substr in self.args.model for substr in {'Cycle'}):
                                outputs = self.model(batch_x, batch_cycle)
                            elif any(substr in self.args.model for substr in
                                    {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF'}):
                                outputs = self.model(batch_x)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if any(substr in self.args.model for substr in {'Cycle'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()

                    pred = outputs
                    true = batch_y

                    preds.append(pred)
                    trues.append(true)
                    inputx.append(batch_x.detach().cpu().numpy())
                    if i % 20 == 0:
                        input = batch_x.detach().cpu().numpy()
                        gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                        np.savetxt(os.path.join(folder_path, str(i) + '.txt'), pd)
                        np.savetxt(os.path.join(folder_path, str(i) + 'true.txt'), gt)
                    
                        
        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
            
            
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        # result save
        if save_visualize :
            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            np.save(folder_path + 'pred.npy', preds)


        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f.write('\n')
        f.write('\n')
        f.close()

        if self.args.model == "iTransformers" and save_visualize:
            attention_weights = np.concatenate(attention_weights, axis=0)
            attention_weights = np.array(attention_weights)

            attention_weights = np.average(attention_weights, axis=0)
            attention_weights = np.average(attention_weights, axis=0)
            
            plt.gca().get_legend().remove()
            np.save(folder_path + 'attention_weights.npy', attention_weights)
            plt.imshow(attention_weights, cmap="hot")
            plt.colorbar()
            plt.savefig(folder_path + 'attention_weights_heat_map.png')
        return mse, mae


    def predict(self, setting, save_visualize=False, load=False):
        
     
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
            
            
        preds = []
        self.model.eval()
        with torch.no_grad():
            if 'Cycle' not in self.args.model:
                
                for i, data in enumerate(pred_loader):
                    
                    
                    batch_x, batch_y, batch_x_mark, batch_y_mark = data
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float()
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.model in self.non_transformer_model_list:
                                
                                outputs = self.model(batch_x)
                            else:
                                if self.args.output_attention and self.args.model == "iTransformers":
                                    outputs, attention_weight = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                    
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.model in self.non_transformer_model_list:
                            
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention and self.args.model == "iTransformers":
                                outputs, attention_weight = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                    
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
                    pred = outputs.detach().cpu().numpy()
                    preds.append(pred)
                    
            else:
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(pred_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float()
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    batch_cycle = batch_cycle.int().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                        batch_y.device)
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if any(substr in self.args.model for substr in {'Cycle'}):
                                outputs = self.model(batch_x, batch_cycle)
                            elif any(substr in self.args.model for substr in
                                    {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF'}):
                                outputs = self.model(batch_x)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if any(substr in self.args.model for substr in {'Cycle'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    pred = outputs.detach().cpu().numpy() 
                    preds.append(pred)
                        
                    
                    
        preds = np.array(preds)
        preds = np.concatenate(preds, axis=0)
        if (pred_data.scale):
            preds = pred_data.inverse_transform(preds)
        
        # result save
        if save_visualize :
            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            np.save(folder_path + 'real_prediction.npy', preds)
            pd.DataFrame(np.append(np.transpose([pred_data.future_dates]), preds[0], axis=1), columns=pred_data.cols).to_csv(folder_path + 'real_prediction.csv', index=False)

            if self.args.model == "iTransformers" :
                np.save(folder_path + 'attention_weights.npy', self.model.attention_weights)
                plt.imsave(folder_path + 'attention_weights_heat_map.png', 
                        self.model.attention_weights, cmap='hot')
        return
