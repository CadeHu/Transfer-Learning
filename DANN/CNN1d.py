import torch.nn as nn

#  m = nn.Conv1d(in_channels=Ci, out_channels=Co, kernel_size=K, stride=s) = output[N,Co,Lo]

#  正常卷积层输出大小计算方法：
#  其中Lo = [(Li-(K-1)+1)/s]+1 向下取整

class CNN1d(nn.Module):
    def __init__(self,n_hidden=160 ,n_class=12):

        super(CNN1d, self).__init__()

        self.layer1 = nn.Sequential(
            #  [480,1,2048]  -> [480,32,256]
            nn.Conv1d(in_channels=1, out_channels=40, kernel_size=32, stride=2),
            nn.BatchNorm1d(40),
            nn.ReLU(),
            #  [480,32,256]  -> [480,32,128]
            nn.MaxPool1d(2)
        )

        self.layer2 = nn.Sequential(
            #  [480,32,128]  -> [480,8,125]
            nn.Conv1d(in_channels=40, out_channels=20, kernel_size=16, stride=2),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            # [480,8,125] -> [480,8,62]
            nn.MaxPool1d(2)
        )

        self.fc1 = nn.Linear(2380 ,n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, src,tar):
        # input.shape:(N,1,n_input)
        x_src = self.layer1(src)
        x_src = self.layer2(x_src)
        # print(x_src.size())
        x_src_mmd = x_src.view(x_src.size(0), -1)
        # 这里为了设置全连接层的输入神经元个数，故需要展示平坦层的特征数（=全连接层输入神经元个数）
        # print(x_src_mmd.size(1))

        x_tar = self.layer1(tar)
        x_tar = self.layer2(x_tar)
        x_tar_mmd = x_tar.view(x_tar.size(0), -1)


        y_src = self.fc1(x_src_mmd)
        y_src = self.fc2(y_src)
        y_src = self.sigmoid(y_src)

        return y_src,x_src_mmd,x_tar_mmd

