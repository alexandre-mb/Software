close all
clear all
clc

% Preparing images for DeepLearning
count = 0;
origImg = imread('IDRiD_01_OD.tif');
[m,n,p] = size(origImg);
for j = 1:(n/244)
    for i = 1:(m/244)
        count = count+1;
        offset_i = i*244;
        offset_j = j*244;
        A = origImg((offset_i-243):offset_i,(offset_j-243):offset_j,:);
        txt = sprintf('C:/Users/Alexandre/Desktop/Software/Preparing Imagens/zIDRiD_01_pt%d.jpg',count);
        imwrite(A,txt);
    end
end