import torch
import torch.nn as nn

'''
m = nn.Conv1d(in_channels=Ci, out_channels=Co, kernel_size=K, stride=s)使用介绍：

对于输入网络的数据:

input=torch.randn(N,Ci,Li)

网络参数的结构：

Weight[Co,Ci,K]

Bias[Co]

网络输出与参数和输入之间的关系：

output[N,Co,Lo] , 其中Lo = [(Li-(K-1)+1)/s]+1 向下取整

Ni =  0,1,2...N-1,  Coi = 0,1,2...Co-1,  

output[Ni][Coi][Loi] = m.bias[Coi]
for i in range(m.weight.size(2)):
    for j in range(m.weight.size(1)):
        output[Ni][Coi][Loi] += m.weight[Coi][j][i] * input[Ni][j][i]

这里变换后的每个样本的特征向量（一行）output[Ni][Coi][Lo]还有待弄清楚。
'''

m = nn.Conv1d(in_channels=2, out_channels=5, kernel_size=8, stride=1)
print(m)
print("==================")
input = torch.randn(10, 10, 2)   # 1 为样本个数，3为特征向量长度 ， 2 为 输入通道数
input = input.permute(0, 2, 1)
print("样本输入大小：{}".format(input.shape))
print("==================")

output = m(input)
print("权值大小：{}".format(m.weight.shape))
print("偏置：{}".format(m.bias.shape))
print("==================")
print("输出大小：{}".format(output.shape))
print("==================")
###  展示样本输出个别元素 及其与 输入和参数之间的关系
Ni = 9
Coi = 1
Loi = 0
print(output[Ni][Coi][Loi])
res = m.bias[Coi]
for i in range(m.weight.size(2)):
    for j in range(m.weight.size(1)):
        res += m.weight[Coi][j][i] * input[Ni][j][i]
print(res)

#
# N = 2 , K =  kernel_size
# Ci = 2, Co = 5
# Li = 5, Lo = ((5-(2-1)-1)/2) + 1 = 2
# input[N,Ci,Li] = [2,2,5]
# output[N,Co,Lo] = [2,5,2]
# weight[Co,Ci,K] = [5,2,2]
# bias [5]
# output[0][0][0] = (m.weight[0][0][0] * input[0][0][0] + m.weight[0][0][1] * input[0][0][1]
#                    m.weight[0][1][0] * input[0][1][0] + m.weight[0][1][1] * input[0][1][1] + m.bias[0])