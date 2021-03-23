import torch.nn as nn
import mmd
#  m = nn.Conv1d(in_channels=Ci, out_channels=Co, kernel_size=K, stride=s) = output[N,Co,Lo]

#  正常卷积层输出大小计算方法：
#  其中Lo = [(Li-(K-1)+1)/s]+1 向下取整

class CNN1d(nn.Module):
    def __init__(self,n_hidden=60 ,n_class=3):

        super(CNN1d, self).__init__()

        self.featureCap = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=30, kernel_size=16, stride=2),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.MaxPool1d(2),
            #  [~,20,256]  -> [480,32,128]
            nn.Conv1d(in_channels=30, out_channels=20, kernel_size=8, stride=2),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.MaxPool1d(2)
            # [480,8,125] -> [480,8,62]
        )

        # self.fc1 = nn.Linear(680 ,n_hidden)
        self.fc1 = nn.Linear(560 ,n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_class)
        # self.sigmoid = nn.Sigmoid()



    def forward(self, src, tar):
        loss = 0
        x_src = self.featureCap(src)

        x_src_mmd = x_src.view(x_src.size(0), -1)
        # 这里为了设置全连接层的输入神经元个数，故需要展示平坦层的特征数（=全连接层输入神经元个数）
        # print(x_src_mmd.size(1))

        if self.training == True:
            x_tar = self.featureCap(tar)

            x_tar_mmd = x_tar.view(x_tar.size(0), -1)
            #loss += mmd.mmd_rbf_accelerate(source, target)
            loss += mmd.mmd_rbf_noaccelerate(x_src_mmd, x_tar_mmd)

        y_src = self.fc1(x_src_mmd)
        y_src = self.fc2(y_src)
        #target = self.cls_fc(target)

        return y_src, loss


