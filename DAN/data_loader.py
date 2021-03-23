import torch
from torch.utils.data import TensorDataset,DataLoader
import scipy.io as io


def load_training(root_dir,domain,batch_size):
    df_data = io.loadmat(root_dir)
    datas = df_data[domain]
    # 将数据转变成 Tensor 类型
    x, y= torch.from_numpy(datas[:, 1:]).type(torch.FloatTensor).cuda(), torch.from_numpy(datas[:, 0:1]).type(torch.LongTensor).cuda()

    data = x.unsqueeze(1) # 将数据增加维度变为 [batch_size, 1, n_length]
    label= y.squeeze()    # 将标签的维度减少 [batch_size,1] -> [batch_size]
    # 这里这样做的原因:
    # 1.y_pred是classes的 OneHot编码方式
    # 2.label必须是[0, #classes] 区间的一个数字

    data_set = TensorDataset(data, label) # 输入变量在前，输出变量在后
    # 采用 DataLoader 封装 data_set 即可得到能够用于训练神经网络的 data_loader
    data_loader = DataLoader(dataset=data_set ,batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    return data_loader

def load_testing(root_dir,domain,batch_size):
    df_data = io.loadmat(root_dir)
    datas = df_data[domain]
    # 将数据转变成 Tensor 类型
    x, y = torch.from_numpy(datas[:, 1:]).type(torch.FloatTensor).cuda(), torch.from_numpy(datas[:, 0:1]).type(torch.LongTensor).cuda()

    data = x.unsqueeze(1)  # 将数据增加维度变为 [batch_size, 1, n_length]
    label = y.squeeze()  # 将标签的维度减少 [batch_size,1] -> [batch_size]

    data_set = TensorDataset(data, label)  # 输入变量在前，输出变量在后
    # 采用 DataLoader 封装 data_set 即可得到能够用于训练神经网络的 data_loader
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    return data_loader
# #
if __name__ == '__main__':
    src_dir = './source_datasets_Eq_shuffle.mat'
    tar_dir = './target_datasets_Eq_shuffle.mat'

    src_name = "source_data_train"
    tgt_train_name = "target_data_train"
    tgt_test_name = "target_data_test"
    torch.manual_seed(1)
    data_src = load_training(root_dir=src_dir, domain=src_name, batch_size=30)
    tgt_train_loader = load_training(tar_dir, tgt_train_name, batch_size=30)
    tgt_test_loader = load_testing(tar_dir, tgt_test_name, batch_size=30)

    for i_batch, batch_data in enumerate(data_src):
        if i_batch < 1:
            data, label = batch_data
            print(i_batch)  # 打印batch编号
            print(label)
        else:

            break
    print("======================")
    for i_batch, batch_data in enumerate(tgt_train_loader):
        if i_batch < 1:
            data, label = batch_data
            print(i_batch)  # 打印batch编号
            print(label)
        else:

            break

    # for i_batch, batch_data in enumerate(tgt_test_loader):
    #     if i_batch < 4:
    #         data, label = batch_data
    #         print(i_batch)  # 打印batch编号
    #         print(label)
    #     else:
    #
    #         break