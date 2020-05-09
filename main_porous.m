function [] =  main_porous()
%%
% This function implements the whole pipeline of classification of porous
% models. The training data set is in the directory `Sample_training', and
% the test data set is in the directory `Sample_test'.
% The whole pipeline includes computing barcodes, generating feature
% vectors of PRF, dimension reduction and classification.
%

% Independence:
% Python 3.5 or more
% Matlab: Parallel computing toolbox, drtoolbox
% 
%%
fprintf('Classification of Porous Models\n');

drReq = input('Is the dimension reduction needed when dealing with feature vectors? (Y/N):','s');
if strcmp(drReq,'y') || strcmp(drReq,'Y') || strcmp(drReq,'Yes') || strcmp(drReq,'YES') || strcmp(drReq,'yes')
    dr_flag = 1; 
elseif strcmp(drReq,'N') || strcmp(drReq,'n') || strcmp(drReq,'No') || strcmp(drReq,'NO') || strcmp(drReq,'no')
    dr_flag = 0;
else
    fprintf('Error: Invalid string for DR requirement\n');
end


% -----------Computing Barcodes from filtration of V-R complex --------------
trBCdir = 'BC_training';
trBCpath = ['./data1/' trBCdir];
if ~exist(trBCpath,'dir')
    mkdir(trBCpath)
    fprintf('Start computing PDs from training data\n');
    !python BC_compute.py ./data1/Sample_training ./data1/BC_training
    % If some error occurs here, please copy the command above and delete
    % the character '!' in the command console of the current path
    system(['python PD_compute.py ', trSampath, ' ',trBCpath]);
    fprintf('Finish computing PDs from training data\n');
end

tsBCdir = 'BC_test';
tsBCpath = ['./data1/' tsBCdir];
if ~exist(tsBCpath,'dir')
    mkdir(tsBCpath)
    fprintf('Start computing PDs from test data\n');
    !python BC_compute.py ./data1/Sample_test ./data1/BC_test
    % If some error occurs here, please copy the command above and delete
    % the character '!' in the command console of the current path
    fprintf('Finish computing PDs from test data\n');
end

% -----------Generating PRF from Barcodes and Computing Feature vectors--------------
% Parameter of Haar decomposition


haar_n = 5;
sam_num = 8;

% training data
trVecpath = fullfile(cd, 'data1','vec_training');   
trBCfilePath = [trBCpath '/*.txt'];
trBCfiles = dir(trBCfilePath);
trBCfile_num = length(trBCfiles);


% test data
tsVecpath = fullfile(cd, 'data1','vec_test');  
tsBCfilePath = [tsBCpath '/*.txt'];
tsBCfiles = dir(tsBCfilePath);
tsBCfile_num = length(tsBCfiles);

% To evaluate the range of PRF
% To consider the function in [0,bd_xy]^2
dmax = 0;
for i =1:trBCfile_num
    file_name = trBCfiles(i).name;
    fullpath = fullfile(trBCpath, file_name);
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
    for i =1:trBCfile_num
        file_name = trBCfiles(i).name;
        fullpath = fullfile(trBCpath, file_name);
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
        % fprintf([file_name ' finished \n']);
        fclose(fileid);
    end
    fprintf('Finish computing feature vectors of training data \n');
end

% test data
if ~exist(tsVecpath,'dir')
    mkdir(tsVecpath);
    fprintf('Start computing feature vectors of test data \n');
    for i =1: tsBCfile_num
        file_name = tsBCfiles(i).name;
        fullpath = fullfile(cd, tsBCpath, file_name);
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
        % fprintf([file_name ' finished \n']);
        fclose(fileid);
    end
end

% -----------To extract labels from names of file--------------
% labels of training data
trVecfilePath = fullfile(trVecpath,'*.txt');
trVecfiles = dir(trVecfilePath);
trVecfiles_num = length(trVecfiles);

label_tr = zeros(trVecfiles_num,1);
expr1 = '_(\d)_';
for i = 1:trVecfiles_num
   file_name = trVecfiles(i).name;
   [sI,eI] = regexpi(file_name,expr1);
   label_tr(i) = fix(str2double(file_name(sI+1:eI-1)));
end

wfile = fullfile(cd,'data1','training_labels.txt');
fileid = fopen(wfile,'w');
for i = 1:trVecfiles_num
   fprintf(fileid, '%d\n ',label_tr(i));
end
fclose(fileid);

% labels of test data
tsVecfiles = dir([tsVecpath '/*.txt']);
tsVecfiles_num = length(tsVecfiles);

label_ts = zeros(tsVecfiles_num,1);
expr2 = '-(\d)-';
for i = 1:tsVecfiles_num
   file_name = tsVecfiles(i).name;
   [sI,eI] = regexpi(file_name,expr2);
   label_ts(i) = fix(str2double(file_name(sI+1:eI-1)));
end
wfile = fullfile(cd,'data1','test_labels.txt');
fileid = fopen(wfile,'w');
for i = 1:tsVecfiles_num
   fprintf(fileid, '%d\n ',label_ts(i));
end
fclose(fileid);

% ----------- Model of Dimension Reduction--------------
if dr_flag
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
    wdir_name1 = fullfile(cd,'data1', 'dr_training');
    if ~exist(wdir_name1,'dir')
        mkdir(wdir_name1);
    end
    wdir_name2 = fullfile(cd,'data1', 'dr_test');
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
    fprintf('Finish dimension reduction\n');
end

%-----------------Classification-------------------------
fprintf('Classifying\n');
if dr_flag
    if exist(wdir_name1,'dir') && exist(wdir_name2,'dir')
       trDrpath = './data1/dr_training';
       tsDrpath = './data1/dr_test';
       trLabel = './data1/training_labels.txt';
       tsLabel = './data1/test_labels.txt';
       system(['python classifiers.py ' trDrpath ' ' tsDrpath ' ' trLabel ' ' tsLabel]);
    else
        fprintf('dr_training or dr_test: no such directory.\n');
    end
else
    if exist(trVecpath,'dir') && exist(tsVecpath,'dir')
       trDrpath = './data1/vec_training';
       tsDrpath = './data1/vec_test';
       trLabel = './data1/training_labels.txt';
       tsLabel = './data1/test_labels.txt';
       system(['python classifiers.py ' trDrpath ' ' tsDrpath ' ' trLabel ' ' tsLabel]);
    else
        fprintf('vec_training or vec_test: no such directory.\n');
    end
end