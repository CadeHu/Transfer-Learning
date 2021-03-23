clear all;
close all;
clc;

%% ���� Դ������
sun_duanchi = load('D:/������/���ݼ�/���ǳ���������/̫���ֶϳ�/xcx_sungear_duanchi_1200r_12800Hz_JZ57_1.TXT');
source_datas.sun_duanchi = sun_duanchi ;
sun_quechi = load('D:/������/���ݼ�/���ǳ���������/̫����ȱ��/xcx_sungearquechi_12800Hz_1200r_JZ53#1.TXT');
source_datas.sun_quechi = sun_quechi ;
sun_liewen = load('D:/������/���ݼ�/���ǳ���������/̫��������/xcx_sungearliewen_1200r_12.8KHz_60s7#1.TXT');
source_datas.sun_liewen=sun_liewen;

%% ���� Ŀ��������

sun_duanchi_1 = load('D:\������\���ݼ�\���ǳ���������\���Ƕϳ�\xcx_xxlduanchi_12800Hz_1200r_JZ53#1.TXT');
target_datas.planet_duanchi = sun_duanchi_1 ;

sun_quechi_1 =  load('D:\������\���ݼ�\���ǳ���������\����ȱ��\xcx_xxquechi_1200r_12800Hz_10s7#1.TXT');
target_datas.planet_quechi = sun_quechi_1 ;
sun_liewen_1 = load('D:\������\���ݼ�\���ǳ���������\��������\xcx_xxliewen_1200r_12800Hz_10s7#1.TXT');
target_datas.planet_liewen= sun_liewen_1;


%% ����ֲ�����
clear sun_duanchi_1 sun_quechi_1 sun_liewen_1;
clear sun_duanchi sun_quechi sun_liewen ;
clear normal planet_duanchi planet_quechi planet_liewen ring_duanchi ring_quechi ring_liewen ;
%% ���ò���
ratio =0.5;
split_ratio = 0.3;    %ѵ�����Ͳ��Լ��Ļ��ֱ���
rand = 1;
len_sample  =640; % ÿ�����ݳ���
overlop=ratio*len_sample;  % ÿ���ƶ�overlop����
num_sample=floor(floor(128000/overlop)-1/ratio+1);
(num_sample-1)*overlop+len_sample >128000

%% ����Դ�����ݼ���Ŀ�������ݼ�
[source_data_train,~,~] = generate(source_datas,num_sample,overlop, len_sample,rand,0);
[~,target_data_train,target_data_test] = generate(target_datas,num_sample,overlop, len_sample,rand,split_ratio);
save('../DAN/source_datasets.mat','source_data_train');
save('../DAN/target_datasets.mat','target_data_train','target_data_test');
