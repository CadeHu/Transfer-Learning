function [source_data,train_data,test_data] = generate(datas,num_sample,overlop, len_sample,rand,split_ratio)
%输入参数 ：
%     datas: 原始数据                         num_sample: 每类数据个数      num_class：数据种类
%     overlop：相邻数据移动点数      len_sample:每段数据长度
%     split_ratio: 训练集占比              rand_unbalance=0 or 1：   1则生成不平衡数据，0则生成平衡数据
%     
% 输出参数：
%      source_data: 整个数据集
%      train_data：训练集
%      test_data:  测试集
%
    names = fieldnames(datas);
    num_class=length(names);
    new_datas=zeros(num_sample ,len_sample,num_class);
    
    for j =1:num_class
        Temp=datas.(names{j});
        for i=1:num_sample 
            new_datas(i,:,j)=Temp((i-1)*overlop+1:(i-1)*overlop+len_sample);
        end
    end
     
    %% 构建数据集source_data  
    L = len_sample ;    %设置数据长度L
    %% 构建测试集和训练集
    tr_num = round(num_sample*split_ratio);
    te_num = num_sample-tr_num;
    %% 
    for i=1:num_class
        source_data(1+num_sample*(i-1):num_sample*i , 1) = i-1;
        source_data(1+num_sample*(i-1):num_sample*i , 2:L+1) = zscore(new_datas(:,1:L,i));
        train_data(1+tr_num *(i-1) : tr_num *i,1)= i-1;
        train_data(1+tr_num*(i-1):tr_num*i ,  2:L+1)=zscore(new_datas(1:tr_num,1:L,i));
        test_data(1+te_num *(i-1) : te_num *i,1)= i-1;
        test_data(1+te_num*(i-1):te_num*i ,  2:L+1)=zscore(new_datas(tr_num+1:end,1:L,i));
    end
    %% 是否打乱各类之间的顺序
    if rand ==1
        source_data = source_data(randperm(num_sample*num_class),:);   
        train_data = train_data(randperm(tr_num*num_class),:);
        test_data = test_data(randperm(te_num*num_class),:);
    end
    
    
return 