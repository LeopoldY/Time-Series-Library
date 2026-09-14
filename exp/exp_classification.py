from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

warnings.filterwarnings('ignore')

import torch.nn.functional as F # 确保头部导入了 F

# ================= 核心：Focal Loss 定义 =================
class BinaryFocalLoss(nn.Module):
    """
    Focal Loss: 专注于难分样本，解决极度不平衡问题
    公式: FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        # alpha: 平衡正负样本权重 (0.25是论文默认值，正样本少时可适当调大，如0.75)
        # gamma: 关注难分样本 (2.0是标准值，越大越关注难样本)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # 针对二分类任务，输入 logits 维度通常为 [Batch, 1] 或 [Batch]
        # 如果是多分类 (c_out=2) 使用 Softmax 输出，这里需要适配 CrossEntropy 的逻辑
        
        # 这里的实现是针对 CrossEntropy (Softmax) 的形式
        # inputs: [Batch, C], targets: [Batch]
        
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # 动态计算 alpha
        # 如果 target=1, alpha_t = alpha
        # 如果 target=0, alpha_t = 1-alpha
        # 但标准的 CrossEntropy Focal Loss 通常简化为只对 loss 加权
        
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
             # 手动根据 label 施加 alpha
             # 假设 1 是正类
             alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
             focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
# =======================================================


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # 使用 Focal Loss
        # alpha=0.75 表示给予正样本(Label=1)更高的权重 (0.75 vs 0.25)
        # gamma=2.0 表示降低简单样本的权重，专注于难分样本
        criterion = BinaryFocalLoss(alpha=0.7, gamma=2.0)
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                # ==================== 【关键修复】 ====================
                # 1. 之前代码用了 .cpu() 导致和 GPU 上的 criterion 冲突
                # 2. 直接在 GPU 上计算 Loss
                # 3. 使用 .item() 取值，避免后续 np.average 报错
                # ====================================================
                
                # 修正前 (您的报错代码): 
                # pred = outputs.detach().cpu()
                # loss = criterion(pred, label.long().squeeze().cpu())

                # 修正后:
                loss = criterion(outputs, label.long().squeeze())
                
                # 必须加 .item()，否则还是会报 numpy 转换错误
                total_loss.append(loss.item())

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        # 后处理逻辑保持不变 (在最后统一转 CPU 计算指标)
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        
        # Softmax + Argmax (针对二分类 c_out=2)
        probs = torch.nn.functional.softmax(preds, dim=1) 
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
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
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        # ==========================================================
        # 核心修改：计算多维度指标
        # ==========================================================
        print("\n" + "="*20 + " Test Results " + "="*20)

        # 【关键修复】必须加 .cpu() 才能转 numpy
        
        # 1. 打印详细报告 (包含每一类的 P/R/F1)
        # digits=4 保证保留4位小数
        report = classification_report(trues, predictions, digits=4)
        print(report)
        
        # 2. 计算核心指标 (关注 label=1 即故障类)
        # binary模式下，只计算 positive label (1) 的指标
        acc = accuracy_score(trues, predictions)
        precision = precision_score(trues, predictions, average='binary', pos_label=1)
        recall = recall_score(trues, predictions, average='binary', pos_label=1)
        f1 = f1_score(trues, predictions, average='binary', pos_label=1)
        
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print("="*54)

        # 3. 保存结果到 CSV
        result_dict = {
            'Timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'Model': self.args.model,
            'Seq_Len': self.args.seq_len,
            'Accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'Exp_ID': setting
        }
        
        summary_file = 'fault_classification_results.csv'
        df_new = pd.DataFrame([result_dict])
        
        if os.path.exists(summary_file):
            df_new.to_csv(summary_file, mode='a', header=False, index=False)
        else:
            df_new.to_csv(summary_file, mode='w', header=True, index=False)
        
        print(f"Metrics saved to {os.path.abspath(summary_file)}")

        # result save
        folder_path = './results_classification/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('accuracy:{}'.format(accuracy))
        file_name='result_classification.txt'
        f = open(os.path.join(folder_path,file_name), 'a')
        f.write(setting + "  \n")
        f.write('accuracy:{}'.format(accuracy))
        f.write('\n')
        f.write('\n')
        f.close()
        return
