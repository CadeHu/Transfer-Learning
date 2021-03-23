from __future__ import print_function

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import MyNet
import data_loader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training settings
Classes = 4
batch_size = 100
lr = 0.01
# 设置随机种子
seed = 16
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# momentum = 0.9
momentum = 2
l2_decay = 5e-4
# l2_decay = 1e-4

# src_dir = './source_datasets_UN.mat'
# tar_dir = './target_datasets_UN.mat'

src_dir = './source_datasets_Eq_shuffle_HHT.mat'
tar_dir = './target_datasets_Eq_shuffle_HHT.mat'

src_name = "source_data_train"
tgt_train_name = "target_data_train"
tgt_test_name = "target_data_test"

src_loader = data_loader.load_training(src_dir, src_name, batch_size)
tgt_train_loader = data_loader.load_training(tar_dir, tgt_train_name, batch_size)
tgt_test_loader = data_loader.load_testing(tar_dir, tgt_train_name, batch_size)

src_dataset_len = len(src_loader.dataset)
tgt_train_dataset_len = len(tgt_train_loader.dataset)
tgt_test_dataset_len = len(tgt_test_loader.dataset)
src_loader_len = len(src_loader)
tgt_loader_len = len(tgt_train_loader)

print('源域训练集个数:%d'%src_dataset_len)
print('目标域训练集个数:%d'%tgt_train_dataset_len)
print('源域训练集个数:%d'%tgt_test_dataset_len)

print("源域批次个数：%d"%src_loader_len)
print("目标域批次个数：%d"%tgt_loader_len)

# 设置迭代次数iteration和训练完一次所有数据（log_interval批）展示一下损失值
log_interval = src_loader_len
iteration = 1000*src_loader_len
RESULT_Test = []


def train(model):
    src_iter = iter(src_loader)
    tgt_iter = iter(tgt_train_loader)
    correct = 0
    for i in range(1, iteration + 1):
        model.train()
        # LEARNING_RATE = lr
        LEARNING_RATE = lr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        if (i - 1) % (10*log_interval) == 0:
            print('learning rate{: .4f}'.format(LEARNING_RATE))

        # optimizer = torch.optim.SGD([
        # {'params': model.sharedNet.parameters()},
        # {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
        # ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)

        optimizer = torch.optim.SGD([
            {'params': model.featureCap.parameters()},
            {'params': model.fc2.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay,nesterov=True)

        # optimizer = torch.optim.SGD(
        #     model.parameters(), lr=LEARNING_RATE , momentum=momentum, weight_decay=l2_decay, nesterov=True)

        # optimizer = torch.optim.Adamax([
        #     {'params': model.featureCap.parameters()},
        #     {'params': model.fc2.parameters(), 'lr': LEARNING_RATE},
        # ], lr=LEARNING_RATE / 10, weight_decay=l2_decay)

        try:
            src_data, src_label = src_iter.next()
        except Exception as err:
            src_iter = iter(src_loader)
            src_data, src_label = src_iter.next()

        try:
            tgt_data, _ = tgt_iter.next()
        except Exception as err:
            tgt_iter = iter(tgt_train_loader)
            tgt_data, _ = tgt_iter.next()

        if torch.cuda.is_available():
            src_data, src_label = src_data.cuda(), src_label.cuda()
            tgt_data = tgt_data.cuda()
        # print(src_data.shape)
        optimizer.zero_grad()
        src_pred, mmd_loss = model(src_data, tgt_data)

        cls_loss = F.nll_loss(F.log_softmax(src_pred, dim=1), src_label)

        lambd = 2 / (1 + math.exp(-10 * (i) / iteration)) - 1
        # lambd = 10
        loss = cls_loss + lambd * mmd_loss
        loss.backward()
        optimizer.step()

        # 每隔10次显示一次损失值
        if i % log_interval == 0:
            print('Train iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}'.format(
                i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item()))

        # 每隔10*20次计算目标域测试集准确率，并更新迁移学习的最大准确率
        if i % (log_interval * 20) == 0:
            t_correct = test(model)
            if t_correct > correct:
                correct = t_correct
            print('src: {} to tgt: {} max correct: {} max accuracy{: .2f}%\n'.format(
                src_name, tgt_train_name, correct, 100. * correct / tgt_test_dataset_len))
            # RESULT_Train.append(100. * correct / tgt_dataset_len)


def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for tgt_test_data, tgt_test_label in tgt_test_loader:

            if torch.cuda.is_available():
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()

            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)

            tgt_pred, mmd_loss = model(tgt_test_data, tgt_test_data)

            test_loss += F.nll_loss(F.log_softmax(tgt_pred, dim=1), tgt_test_label,
                                    reduction='sum').item()  # sum up batch loss

            pred = tgt_pred.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()

    test_loss /= tgt_test_dataset_len
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        tgt_test_name, test_loss, correct, tgt_test_dataset_len, 100. * correct / tgt_test_dataset_len))

    RESULT_Test.append(100. * correct / tgt_test_dataset_len)
    return correct


if __name__ == '__main__':
    # model = models.DANNet(num_classes=31)
    model = MyNet.CNN1d(n_hidden=120, n_class=Classes)
    print(model)
    model = model.to(DEVICE)
    train(model)
    res_test = np.asarray(RESULT_Test)


    # np.savetxt('test_acc_UN.csv', res_test, fmt='%.6f ', delimiter=',')
    np.savetxt('test_acc_Eq_HHT.csv', res_test, fmt='%.6f ', delimiter=' ')
