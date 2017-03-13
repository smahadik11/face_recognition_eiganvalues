 allFiles = dir('C:\Users\Sudha\Documents\MATLAB\csc872\ALL\*.tif');  
 no_files = length(allFiles);%number of files
 IM = zeros(32*32, no_files);
 for i = 1 : no_files
     input = strcat('C:\Users\Sudha\Documents\MATLAB\csc872\ALL\',allFiles(i).name);
     image = imread(input);%read image
     IM(:, i) = reshape(double(image), [], 1);
 end
M = mean(IM,2);
S = cov(IM');
% Calculate eigenvalues of COV
[V, D] = eig(S);% A*V = V*D.
d = diag(D);
sum_diag = sum(d);
cdf(1) = d(1) / sum_diag;
for i=2:32
    cdf(i) = cdf(i-1) + d(i)/ sum_diag;
end
plot(cdf);
title('cumulative distribution','fontsize',20);

[B,I] = sort(diag(D), 'descend');%eignvector sorting
K = 20;
W = zeros(32*32, K);
for j=1:K
    W(:, j) = V(:, I(j));%selected top K eignvectors resulting in a face model
end
 
%training
FA = dir('D:\Sudha_Project\Images\FA\*.TIF'); 
count_A = length(FA);
y = zeros(K, count_A);
for index=1:count_A
   file = FA(index).name;
   file_name  = strcat('D:\Sudha_Project\Images\FA\', file);   
   IM_FA = imread(file_name);
   y(:,index) = W' * (reshape(double(IM_FA(:)),[],1) - M);%construct a DB for known faces %Ci
end

%testing
files_FB = dir('D:\Sudha_Project\Images\FB\*.TIF'); 
count_B = length(files_FB);
for no = 1:count_B
    file_B = files_FB(no).name;
    file_name_B = strcat('D:\Sudha_Project\Images\FB\', file_B); 
    z = reshape(double(imread(file_name_B)), [], 1);
    yz = W'*(z-M);%test face projected to model y , dz 
    diff = zeros(1, count_A);
    for p=1:count_A
        diff(p) = norm(yz - y(:, p));
        [sortedDistValues, sortedDistIndices] = sort(diff, 'ascend');
    end
    disp(strcat('Best Match: D:\Sudha_Project\Images\FA\', FA(sortedDistIndices(1)).name));
    disp(strcat('Distance from test image: ', int2str(sortedDistValues(1))));
    figure;
    subplot(1, 2, 1);
    imshow(file_name_B);
    title('test face');
    subplot(1, 2, 2);
    imshow(strcat('D:\Sudha_Project\Images\FA\', FA(sortedDistIndices(1)).name));
    title('matched face');
end

