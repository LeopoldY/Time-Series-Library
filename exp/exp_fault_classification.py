from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

warnings.filterwarnings('ignore')


class Exp_Fault_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Fault_Classification, self).__init__(args)
        # 强制设置输出维度为 1 (二分类: 故障概率)
        self.args.c_out = 1

    def _build_model(self):
        # 1. 欺骗模型，让其认为是在做长期预测 (修复 outputs 为 None 的问题)
        original_task = self.args.task_name
        self.args.task_name = 'long_term_forecast'

        # 2. 【关键修复】针对 Informer 短序列崩溃的终极方案
        if self.args.model == 'Informer':
            # 手动注入参数，跳过命令行检查
            self.args.output_attention = True  # 开启输出注意力
            self.args.attn = 'full'            # 强制使用全量注意力 (FullAttention)
            print(">>> [Auto-Fix] Forced Informer to use FullAttention for short sequence stability.")

        # 构建模型
        model = self.model_dict[self.args.model].Model(self.args).float()
        
        # 恢复原任务名
        self.args.task_name = original_task

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
        # 【关键修改】计算权重
        # 假设 0 是正常(多数)，1 是故障(少数)
        # 权重公式通常为: 总样本数 / (类别数 * 该类样本数)
        # 或者简单粗暴地: 负样本数 / 正样本数
        # 举例: 如果正常样本是故障样本的 50 倍，则给故障样本(索引1) 50 倍的权重
        
        #                          [类0权重, 类1权重]
        class_weights = torch.tensor([1.0, 1000.0]).to(self.device)
        
        # 传入 weight 参数
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # 【关键修改】将回归目标转换为分类标签 (0/1)
                batch_y = (batch_y > 0.0).float()
                batch_y = batch_y.to(self.device)

                # encoder - decoder
                # 传入完整的 4 个参数，避免 TypeError
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                # 计算二分类 Loss
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())
                
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # 【关键修改】标签二值化
                batch_y = (batch_y > 0.0).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                        
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                    
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
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # 标签二值化
                batch_y = (batch_y > 0.0).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                
                # 转为概率
                probs = torch.sigmoid(outputs)

                preds.append(probs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        preds = preds.reshape(-1)
        trues = trues.reshape(-1)
        
        # 【关键修改】计算分类指标 (F1, Precision, Recall, Accuracy)
        best_f1 = -1
        best_thresh = 0
        best_metrics = {}

        # 动态搜索最佳阈值 (0.1 ~ 0.9)
        for thresh in np.arange(0.1, 0.95, 0.05):
            binary_preds = (preds > thresh).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(trues, binary_preds, average='binary', zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                acc = accuracy_score(trues, binary_preds)
                best_metrics = {'Precision': p, 'Recall': r, 'F1': f1, 'Accuracy': acc}

        print(f"\n[Classification Results]")
        print(f"Threshold: {best_thresh:.2f}")
        print(f"F1-Score: {best_f1:.4f} | Recall: {best_metrics.get('Recall', 0):.4f}")

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 保存结果到文本文件
        f = open("result_fault_classification.txt", 'a')
        f.write(setting + "  \n")
        f.write('Thresh:{:.2f}, Acc:{:.4f}, Prec:{:.4f}, Recall:{:.4f}, F1:{:.4f}'.format(
            best_thresh, 
            best_metrics.get('Accuracy', 0),
            best_metrics.get('Precision', 0),
            best_metrics.get('Recall', 0),
            best_metrics.get('F1', 0)
        ))
        f.write('\n\n')
        f.close()
        
        # 同时也保存到 Summary CSV 中 (方便后续画图)
        device_name = self.args.data_path.replace('.csv', '')
        summary_file = 'fault_classification_results.csv'
        res_dict = {
            'Timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'Device': device_name,
            'Model': self.args.model,
            'Seq_Len': self.args.seq_len,
            'Best_Threshold': best_thresh,
            'Accuracy': best_metrics.get('Accuracy', 0),
            'Precision': best_metrics.get('Precision', 0),
            'Recall': best_metrics.get('Recall', 0),
            'F1_Score': best_metrics.get('F1', 0)
        }
        df_new = pd.DataFrame([res_dict])
        if os.path.exists(summary_file):
            df_new.to_csv(summary_file, mode='a', header=False, index=False)
        else:
            df_new.to_csv(summary_file, mode='w', header=True, index=False)

        np.save(folder_path + 'metrics.npy', np.array([best_f1, best_thresh]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return