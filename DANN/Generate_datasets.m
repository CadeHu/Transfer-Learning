clear all;
close all;
clc;

%% 加载 源域数据
sun_duanchi = load('D:/大论文/数据集/行星齿轮箱数据/太阳轮断齿/xcx_sungear_duanchi_1200r_12800Hz_JZ57_1.TXT');
source_datas.sun_duanchi = sun_duanchi ;
sun_quechi = load('D:/大论文/数据集/行星齿轮箱数据/太阳轮缺齿/xcx_sungearquechi_12800Hz_1200r_JZ53#1.TXT');
source_datas.sun_quechi = sun_quechi ;
sun_liewen = load('D:/大论文/数据集/行星齿轮箱数据/太阳轮裂纹/xcx_sungearliewen_1200r_12.8KHz_60s7#1.TXT');
source_datas.sun_liewen=sun_liewen;

%% 加载 目标域数据

sun_duanchi_1 = load('D:\大论文\数据集\行星齿轮箱数据\行星断齿\xcx_xxlduanchi_12800Hz_1200r_JZ53#1.TXT');
target_datas.planet_duanchi = sun_duanchi_1 ;

sun_quechi_1 =  load('D:\大论文\数据集\行星齿轮箱数据\行星缺齿\xcx_xxquechi_1200r_12800Hz_10s7#1.TXT');
target_datas.planet_quechi = sun_quechi_1 ;
sun_liewen_1 = load('D:\大论文\数据集\行星齿轮箱数据\行星裂纹\xcx_xxliewen_1200r_12800Hz_10s7#1.TXT');
target_datas.planet_liewen= sun_liewen_1;


%% 清除局部变量
clear sun_duanchi_1 sun_quechi_1 sun_liewen_1;
clear sun_duanchi sun_quechi sun_liewen ;
clear normal planet_duanchi planet_quechi planet_liewen ring_duanchi ring_quechi ring_liewen ;
%% 设置参数
ratio =0.5;
split_ratio = 0.3;    %训练集和测试集的划分比例
rand = 1;
len_sample  =640; % 每段数据长度
overlop=ratio*len_sample;  % 每次移动overlop个点
num_sample=floor(floor(128000/overlop)-1/ratio+1);
(num_sample-1)*overlop+len_sample >128000

%% 生成源域数据集和目标域数据集
[source_data_train,~,~] = generate(source_datas,num_sample,overlop, len_sample,rand,0);
[~,target_data_train,target_data_test] = generate(target_datas,num_sample,overlop, len_sample,rand,split_ratio);
save('../DAN/source_datasets.mat','source_data_train');
save('../DAN/target_datasets.mat','target_data_train','target_data_test');
