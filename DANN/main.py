import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import data_loader
# import mmd
import mmd_pytorch
import CNN1d

mmd = mmd_pytorch.MMD_loss( kernel_type='rbf', kernel_mul=2.0, kernel_num=5)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 0.01
MOMEMTUN = 0.05
L2_WEIGHT = 0.003
Classes = 4
N_EPOCH = 100
BATCH_SIZE = [64, 32]
LAMBDA = 0.2
# GAMMA = 10 ^ 3
RESULT_TRAIN = []
RESULT_TEST = []
# log_train = open('log_train_a-w.txt', 'w')
# log_test = open('log_test_a-w.txt', 'w')


# 求出领域之间的最大均值距离
# def mmd_loss(x_src, x_tar):
#     return mmd.mix_rbf_mmd2(x_src, x_tar, [GAMMA])

def train(model, optimizer, epoch, data_src, data_tar):
    total_loss_train = 0
    criterion = nn.CrossEntropyLoss()
    correct = 0
    batch_j = 0
    list_src, list_tar = list(enumerate(data_src)), list(enumerate(data_tar))

    for batch_id, (data, target) in enumerate(data_src):
        # 目标域数据集
        _, (x_tar, y_target) = list_tar[batch_j]
        x_tar, y_target = x_tar.to(DEVICE), y_target.to(DEVICE)
        # 源域数据集
        data, target = data.to(DEVICE), target.to(DEVICE)
        # print(target.size())
        model.train()

        # 源域数据预测输出 y_src  [batch_size,n_classes]
        # 源域数据平坦层 x_src_mmd  [batch_size,]
        # 目标数据平坦层 x_tar_mmd []
        y_src, x_src_mmd, x_tar_mmd = model(data, x_tar)
        # print(y_src.size()) # [480, 12]

      # 前向传播求出预测的值
        loss_c = criterion(y_src, target)
      # 求出领域之间的最大均值距离
        loss_mmd = mmd(x_src_mmd, x_tar_mmd)
        # print(loss_mmd)
        pred = y_src.data.max(1)[1]  # get the index of the max log-probability

        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        loss = loss_c + LAMBDA * loss_mmd
       # 梯度初始化为零
        optimizer.zero_grad()
       # 反向传播求梯度
        loss.backward()
       # 更新所有参数
        optimizer.step()

        total_loss_train += loss.data
        # res_i = 'Epoch: [{}/{}], Batch: [{}/{}], loss: {:.6f}'.format(
        #     epoch, N_EPOCH, batch_id + 1, len(data_src), loss.data
        # )
        batch_j += 1
        if batch_j >= len(list_tar):
            batch_j = 0

    total_loss_train /= len(data_src)
    acc = correct * 100. / len(data_src.dataset)
    res_e = 'Epoch: [{}/{}], training loss: {:.6f}, correct: [{}/{}], training accuracy: {:.4f}%'.format(
        epoch, N_EPOCH, total_loss_train, correct, len(data_src.dataset), acc )
    tqdm.write(res_e)
    # log_train.write(res_e + '\n')
    RESULT_TRAIN.append([epoch, total_loss_train, acc])
    return model


def test(model, data_tar, e):
    total_loss_test = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(data_tar):
            data, target = data.to(DEVICE),target.to(DEVICE)
            model.eval()
            ypred, _, _ = model(data, data)
            loss = criterion(ypred, target)
            pred = ypred.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            total_loss_test += loss.data
        accuracy = correct * 100. / len(data_tar.dataset)
        res = 'Test: total loss: {:.6f}, correct: [{}/{}], testing accuracy: {:.4f}%'.format(
            total_loss_test, correct, len(data_tar.dataset), accuracy
        )
    tqdm.write(res)
    RESULT_TEST.append([e, total_loss_test, accuracy])
    # log_test.write(res + '\n')

if __name__ == '__main__':
    '''
    源域和目标域数据集要求：
        1.每个样本数据长度一致
        2.样本种类一致
    另外，每类样本数据个数可以不一样  ，也就是两个数据集的样本个数可以不一样
    '''
    # src_dir = './source_datasets.mat'
    # tar_dir = './target_datasets.mat'
    src_dir = './source_datasets_Eq_shuffle.mat'
    tar_dir = './target_datasets_Eq_shuffle.mat'
    torch.manual_seed(1)
    data_src = data_loader.load_data(
        root_dir=src_dir, domain='source_data_train', batch_size=BATCH_SIZE[0])

        # root_dir=src_dir, domain='source_data', batch_size=BATCH_SIZE[0])
        # root_dir=src_dir, domain='source_data_device', batch_size=BATCH_SIZE[0])
    data_tar = data_loader.load_test(
        root_dir=tar_dir, domain='target_data_train', batch_size=BATCH_SIZE[1])

        # root_dir=tar_dir, domain='target_data', batch_size=BATCH_SIZE[1])
        # root_dir=tar_dir, domain='target_data_device', batch_size=BATCH_SIZE[1])
    # print(data_src.size())
    model = CNN1d.CNN1d( n_hidden=100, n_class= Classes)
    # 打印输出模型结构
    print(model)
    model = model.to(DEVICE)

    # 定义优化器
    optimizer = optim.Adamax(
        model.parameters(),
        lr=LEARNING_RATE,
        # momentum=MOMEMTUN,
        weight_decay=L2_WEIGHT
    )

    # 迭代训练网络模型
    for e in tqdm(range(1, N_EPOCH + 1)):
        model = train(model=model, optimizer=optimizer,
                      epoch=e, data_src=data_src, data_tar=data_tar)
        test(model, data_tar, e)
    # torch.save(model, 'model_CNN1d_D.pkl')
    # log_train.close()
    # log_test.close()
    # 保存网络输出结果
    res_train = np.asarray(RESULT_TRAIN)
    res_test = np.asarray(RESULT_TEST)
    np.savetxt('res_train_a-w.csv', res_train, fmt='%.6f', delimiter=',')
    np.savetxt('res_test_a-w.csv', res_test, fmt='%.6f', delimiter=',')

