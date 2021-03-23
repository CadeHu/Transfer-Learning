function [source_data,train_data,test_data] = generate(datas,num_sample,overlop, len_sample,rand,split_ratio)
%������� ��
%     datas: ԭʼ����                         num_sample: ÿ�����ݸ���      num_class����������
%     overlop�����������ƶ�����      len_sample:ÿ�����ݳ���
%     split_ratio: ѵ����ռ��              rand_unbalance=0 or 1��   1�����ɲ�ƽ�����ݣ�0������ƽ������
%     
% ���������
%      source_data: �������ݼ�
%      train_data��ѵ����
%      test_data:  ���Լ�
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
     
    %% �������ݼ�source_data  
    L = len_sample ;    %�������ݳ���L
    %% �������Լ���ѵ����
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
    %% �Ƿ���Ҹ���֮���˳��
    if rand ==1
        source_data = source_data(randperm(num_sample*num_class),:);   
        train_data = train_data(randperm(tr_num*num_class),:);
        test_data = test_data(randperm(te_num*num_class),:);
    end
    
    
return 