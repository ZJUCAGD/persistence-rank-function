function [] =  main_GenImg()
%%
% This function implements the whole pipeline of classification of random
% images of Brownian motion. 
% The data is stored in the directory `data2'.
% The whole pipeline includes generating random images, computing barcodes, 
% generating feature, vectors of PRF, dimension reduction and classification.
%

% Independence:
% Python 3.5 or more
% Matlab: Parallel computing toolbox, drtoolbox
% 
%%
fprintf('Classification of random images of Brownian motion\n');

drReq = input('Is the dimension reduction needed when dealing with feature vectors? (Y/N):','s');
if strcmp(drReq,'y') || strcmp(drReq,'Y') || strcmp(drReq,'Yes') || strcmp(drReq,'YES') || strcmp(drReq,'yes')
    dr_flag = 1; 
%     n = input('Please input a positive integer for dimension reduction, from 2 to 100:');
%     n = fix(n);
elseif strcmp(drReq,'N') || strcmp(drReq,'n') || strcmp(drReq,'No') || strcmp(drReq,'NO') || strcmp(drReq,'no')
    dr_flag = 0;
else
    fprintf('Error: Invalid string for DR requirement\n');
end



if ~exist('data2','dir')
    mkdir('data2');
end
% -----------Generating training and test dataset--------------
% Parameters
% To generate random images by Brownian motion
cls = 8;
tr_img_num = 100;
ts_img_num = 20;
N = [100, 250, 150, 250,200, 250, 200, 400];
S = [30, 12, 30, 18, 30, 23, 40, 20];

% training data
tr_imgPath = './data2/imgRd_training';
if ~exist(tr_imgPath,'dir')
    fprintf('Start generating training data\n');
    mkdir(tr_imgPath);
    tr_label_path = './data2/training_labels.txt';
    trls_id = fopen(tr_label_path, 'w');
    for i = 1:cls
        for j = 1:tr_img_num
            img = GenerateRdmBImage(300,N(i),S(i),4,2);
            filename = [num2str(i) '_' num2str(j) '.txt'];
            filepath = fullfile(tr_imgPath,filename);
            save(filepath, 'img', '-ascii');
            fprintf(trls_id,'%d \n',i);
        end
    end
   fclose(trls_id);
end
fprintf('Finish generating training data\n');

% test data
ts_imgPath = './data2/imgRd_test';
if ~exist(ts_imgPath,'dir')
    fprintf('Start generating test data\n');
    mkdir(ts_imgPath);
    ts_label_path = './data2/test_labels.txt';
    tsls_id = fopen(ts_label_path, 'w');
    for i = 1:cls
        for j = 1:ts_img_num
            img = GenerateRdmBImage(300,N(i),S(i),4,2);
            filename = [num2str(i) '_' num2str(j) '.txt'];
            filepath = fullfile(ts_imgPath,filename);
            save(filepath, 'img', '-ascii');
            fprintf(tsls_id,'%d \n',i);
        end
    end
    fclose(tsls_id);
end
fprintf('Finish generating test data\n');

% -----------Computing Barcodes of training and test dataset--------------
% BC of training data
tr_bcPath = './data2/imgBC_training';
if ~exist(tr_bcPath,'dir')
    fprintf('Start generating barcodes of training data\n');
    mkdir(tr_bcPath);
    status = system(['python computeImgBC.py ' tr_imgPath ' ' tr_bcPath]);
    if status ~= 0
        fprintf('Errors occur when computing barcodes. Please parse the following command to the console: python computeImgBC.py ./data2/imgRd_training ./data2/imgBC_training\n');
        return;
    end
    % If some errors occur, parse the following command to the console:
      % python computeImgBC.py ./data2/imgRd_training ./data2/imgBC_training
end
fprintf('Finish generating barcodes of training data\n');

% BC of test data
ts_bcPath = './data2/imgBC_test';
if ~exist(ts_bcPath,'dir')
    fprintf('Start generating barcodes of test data\n');
    mkdir(ts_bcPath);   
    status = system(['python computeImgBC.py ' ts_imgPath ' ' ts_bcPath]);
    if status ~= 0
        fprintf('Errors occur when computing barcodes. Please parse the following command to the console: python computeImgBC.py ./data2/imgRd_test ./data2/imgBC_test\n');
        return;
    end
    % If some errors occur, parse the following command to the console:
      % python computeImgBC.py ./data2/imgRd_test ./data2/imgBC_test
end
fprintf('Finish generating barcodes of test data\n');

% -----------Generating PRF from Barcodes and Computing Feature vectors--------------
% Parameter of Haar decomposition
haar_n = 5;
sam_num = 8;

% training data
trVecpath = fullfile('./data2','vec_training');   
trBCfilePath = [tr_bcPath '/*.txt'];
trBCfiles = dir(trBCfilePath);
trBCfile_num = length(trBCfiles);


% test data
tsVecpath = fullfile('./data2','vec_test');  
tsBCfilePath = [ts_bcPath '/*.txt'];
tsBCfiles = dir(tsBCfilePath);
tsBCfile_num = length(tsBCfiles);

% To evaluate the range of PRF
% To consider the function in [0,bd_xy]^2
dmax = 0;
for i =1:trBCfile_num
    file_name = trBCfiles(i).name;
    fullpath = fullfile(tr_bcPath, file_name);
    data = load(fullpath);    
    death = data(:,2);
    dmax_i = max(death);
    if dmax_i > dmax
        dmax = dmax_i;
    end    
end
bd_xy = fix(10*dmax)/10;
fprintf('bound value: %f\n', bd_xy);

% haar basis 1D
haar_basis = haarBasisM(haar_n,sam_num);

% trainning data
if ~exist(trVecpath,'dir')
    mkdir(trVecpath);
    fprintf('Start computing feature vectors of training data \n');
    parfor i =1:trBCfile_num
        file_name = trBCfiles(i).name;
        fullpath = fullfile(tr_bcPath, file_name);
        data = load(fullpath);   
        birth = data(:,1);
        death = data(:,2);    
        z = PersBettiSur(birth,death,bd_xy,sam_num);
% haar
        lambda_v = haarDecomposition2DFunc(z,haar_basis);
        len = length(lambda_v);    
        wfile = fullfile(trVecpath, file_name);
        fileid = fopen(wfile, 'w');
        for j = 1:len
            fprintf(fileid, '%.20f\n', lambda_v(j));
        end
        fprintf([file_name ' finished \n']);
        fclose(fileid);
    end
end
fprintf('Finish computing feature vectors of training data \n');

% test data
if ~exist(tsVecpath,'dir')
    mkdir(tsVecpath);
    fprintf('Start computing feature vectors of test data \n');
    parfor i =1: tsBCfile_num
        file_name = tsBCfiles(i).name;
        fullpath = fullfile(cd, ts_bcPath, file_name);
        data = load(fullpath);
        birth = data(:,1);
        death = data(:,2);
        z = PersBettiSur(birth,death,bd_xy,sam_num);
    % haar
        lambda_v = haarDecomposition2DFunc(z,haar_basis);
        len = length(lambda_v);    
        wfile = fullfile(tsVecpath, file_name);
        fileid = fopen(wfile, 'w');
        for j = 1:len
            fprintf(fileid, '%.20f\n', lambda_v(j));
        end
        fprintf([file_name ' finished \n']);
        fclose(fileid);
    end
end
fprintf('Finish computing feature vectors of test data \n');

% ----------- Model of Dimension Reduction--------------
if dr_flag    
    trVecfilePath = fullfile(trVecpath,'*.txt');
    trVecfiles = dir(trVecfilePath);
    trVecfiles_num = length(trVecfiles);
    
    tsVecfiles = dir([tsVecpath '/*.txt']);
    tsVecfiles_num = length(tsVecfiles);
    fprintf('Start dimension reduction\n');
    
    % dimension of training data
    file_name = trVecfiles(1).name;
    filepath = fullfile(trVecpath,file_name);
    data = load(filepath);
    len_m = length(data(:,1));

    % To load training data
    M = zeros(trVecfiles_num,len_m);
    for i = 1:trVecfiles_num
        file_name = trVecfiles(i).name;
        filepath = fullfile(trVecpath,file_name);
        data = load(filepath);
        M(i,:) = data(:,1)';
    end
    label_tr = load('./data2/training_labels.txt');
    
    % To estimate reduced dimension via PCA(SVD)
    [~,~,latent] = pca(M);
    s_sum = sum(latent);
    s = 0;
    for k = 1:length(latent) 
        s = s + latent(k);
       if s/s_sum >= 0.99
           break;
       end
    end
    n = k;
    fprintf('Appropriate reduced dimension: %d\n',n);
    
    % dimension of test data
    file_name = tsVecfiles(1).name;
    filepath = fullfile(tsVecpath,file_name);
    data = load(filepath);
    len_ts = length(data(:,1));
    
    % To load training data
    T = zeros(tsVecfiles_num,len_ts); 
    for i = 1:tsVecfiles_num
        file_name = tsVecfiles(i).name;
        filepath = fullfile(tsVecpath,file_name);
        data = load(filepath);
        T(i,:) = data(:,1)';
    end
   
    % Dimension Reduction
    [v_tr, mapping] = Supervised_Isomap(M, label_tr, n, 12, 0.5);
    v_ts=out_of_sample_new(T,mapping);
    % training data
    wdir_name1 = fullfile(cd,'data2', 'dr_training');
    if ~exist(wdir_name1,'dir')
        mkdir(wdir_name1);
    end
    wdir_name2 = fullfile(cd,'data2', 'dr_test');
    if ~exist(wdir_name2,'dir')
        mkdir(wdir_name2);
    end
    for i = 1:trVecfiles_num
        wfile = fullfile(wdir_name1,trVecfiles(i).name);
        fileid = fopen(wfile,'w');
        for j = 1:n 
            fprintf(fileid, '%.20f \n', v_tr(i,j));
        end
        fclose(fileid);
    end

    % test data
    for i = 1:tsVecfiles_num
        wfile = fullfile(wdir_name2,tsVecfiles(i).name);
        fileid = fopen(wfile,'w');
        for j = 1:n 
            fprintf(fileid, '%.20f \n', v_ts(i,j));
        end
        fclose(fileid);
    end    
end
fprintf('Finish dimension reduction\n');
%-----------------Classification-------------------------
fprintf('Classifying\n');
if dr_flag
    if exist(wdir_name1,'dir') && exist(wdir_name2,'dir')
       trDrpath = './data2/dr_training';
       tsDrpath = './data2/dr_test';
       trLabel = './data2/training_labels.txt';
       tsLabel = './data2/test_labels.txt';
       system(['python classifiers.py ' trDrpath ' ' tsDrpath ' ' trLabel ' ' tsLabel]);
    else
        fprintf('dr_training or dr_test: no such directory.\n');
    end
else
    if exist(trVecpath,'dir') && exist(tsVecpath,'dir')
       trDrpath = './data2/vec_training';
       tsDrpath = './data2/vec_test';
       trLabel = './data2/training_labels.txt';
       tsLabel = './data2/test_labels.txt';
       system(['python classifiers.py ' trDrpath ' ' tsDrpath ' ' trLabel ' ' tsLabel]);
    else
        fprintf('vec_training or vec_test: no such directory.\n');
    end
end
