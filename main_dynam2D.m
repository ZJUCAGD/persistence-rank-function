function [] =  main_dynam2D()
%%
% This function implements the whole pipeline of classification of 2D dynamical system data.
% The classification is the same as that in Adams et al.(2017)
% The data set is stored in `data3' in the directory `dynamic2d' in the
% form of matlab .mat format.
% The whole pipeline includes computing barcodes, generating feature
% vectors of PRF, dimension reduction and classification.
%

% Independence:
% Python 3.5 or more
% Matlab: Parallel computing toolbox, drtoolbox
% 
%%
fprintf('Classification of 2D dynamical system data\n');

drReq = input('Is the dimension reduction needed when dealing with feature vectors? (Y/N):','s');
if strcmp(drReq,'y') || strcmp(drReq,'Y') || strcmp(drReq,'Yes') || strcmp(drReq,'YES') || strcmp(drReq,'yes')
    dr_flag = 1; 
elseif strcmp(drReq,'N') || strcmp(drReq,'n') || strcmp(drReq,'No') || strcmp(drReq,'NO') || strcmp(drReq,'no')
    dr_flag = 0;
else
    fprintf('Error: Invalid string for DR requirement\n');
end

if ~exist('data3','dir')
    fprintf('Error: Cannot find data\n');
end
if ~exist('./data3/dynamic2d','dir')
    fprintf('Error: Cannot find data\n');
end
% -----------Generating training and test dataset--------------
if ~exist('./data3/training_data','dir') && ~exist('./data3/test_data','dir')
    system('python dynam2D_BC.py');
end

% -----------Computing Barcodes of training and test dataset--------------
% BC of training data
tr_path = './data3/training_data';
tr_bcPath = './data3/BC_training';
if ~exist(tr_bcPath,'dir')
    fprintf('Start generating barcodes of training data\n');
    mkdir(tr_bcPath);
    status = system(['python BC_compute.py ' tr_path ' ' tr_bcPath]);
    if status ~= 0
        fprintf('Errors occur when computing barcodes. Please parse the following command to the console: python BC_compute.py ./data3/training_data ./data3/BC_training\n');
        return;
    end
    % If some errors occur, parse the following command to the console:
      % python computeImgBC.py ./data2/imgRd_training ./data2/imgBC_training
end
fprintf('Finish generating barcodes of training data\n');

% BC of test data
ts_path = './data3/test_data';
ts_bcPath = './data3/BC_test';
if ~exist(ts_bcPath,'dir')
    fprintf('Start generating barcodes of test data\n');
    mkdir(ts_bcPath);   
    status = system(['python computeImgBC.py ' ts_path ' ' ts_bcPath]);
    if status ~= 0
        fprintf('Errors occur when computing barcodes. Please parse the following command to the console: python BC_compute.py ./data3/test_data ./data3/BC_test\n');
        return;
    end
    % If some errors occur, parse the following command to the console:
      % python computeImgBC.py ./data2/imgRd_test ./data2/imgBC_test
end
fprintf('Finish generating barcodes of test data\n');

% -----------Generating PRF from Barcodes and Computing Feature vectors--------------
% Parameter of Haar decomposition
haar_n = 5;
sam_num =8;

% training data
trVecpath = fullfile('./data3','vec_training');   
trBCfilePath = [tr_bcPath '/*.txt'];
trBCfiles = dir(trBCfilePath);
trBCfile_num = length(trBCfiles);


% test data
tsVecpath = fullfile('./data3','vec_test');  
tsBCfilePath = [ts_bcPath '/*.txt'];
tsBCfiles = dir(tsBCfilePath);
tsBCfile_num = length(tsBCfiles);

% To evaluate the range of PRF
% To consider the function in [0,bd_xy]^2
% interval estimation

file_name = trBCfiles(1).name;
fullpath = fullfile(tr_bcPath, file_name);
X_samp = load(fullpath);

for i =2:trBCfile_num
    file_name = trBCfiles(i).name;
    fullpath = fullfile(tr_bcPath, file_name);
    data = load(fullpath);    
    X_samp = [X_samp;data];
end
X_samp = [X_samp(:,1);X_samp(:,2)];
muhat = expfit(X_samp,0.05);
histogram(X_samp);
bd_xy = -log(0.01)*muhat;
clear('X_samp');
fprintf('bd: %.5f\n',bd_xy);


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
    
    % Estimate of required dimension by PCA
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
    
    label_tr = load('./data3/training_labels.txt');
    
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
    wdir_name1 = fullfile(cd,'data3', 'dr_training');
    if ~exist(wdir_name1,'dir')
        mkdir(wdir_name1);
    end
    wdir_name2 = fullfile(cd,'data3', 'dr_test');
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
       trDrpath = './data3/dr_training';
       tsDrpath = './data3/dr_test';
       trLabel = './data3/training_labels.txt';
       tsLabel = './data3/test_labels.txt';
       system(['python classifiers.py ' trDrpath ' ' tsDrpath ' ' trLabel ' ' tsLabel]);
    else
        fprintf('dr_training or dr_test: no such directory.\n');
    end
else
    if exist(trVecpath,'dir') && exist(tsVecpath,'dir')
       trDrpath = './data3/vec_training';
       tsDrpath = './data3/vec_test';
       trLabel = './data3/training_labels.txt';
       tsLabel = './data3/test_labels.txt';
       system(['python classifiers.py ' trDrpath ' ' tsDrpath ' ' trLabel ' ' tsLabel]);
    else
        fprintf('vec_training or vec_test: no such directory.\n');
    end
end
