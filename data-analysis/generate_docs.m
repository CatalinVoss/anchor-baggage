%
% VisualBaggage
%
% Created by Catalin Voss on 5/11/14
% Copyright (c) All rights reserved
%
% Generate python anchor word recovery input from OpenCV-generated bag of
% visual words (BoW) data.
%

%% Cleanup
clc; clear all; close all;

%% Load CSV

documents = importfile1('/Users/Catalin/Desktop/documents.csv').'; % notice the transpose to form the matrix W as illustrated in the paper
documents = 50*documents; % constant empirically determined by looking at PCA (see below)
documents = round(documents);

%% PCA
data = documents.';

% Check the pairwise correlation between the variables.
C = corr(data,data);

% Perform the principal component analysis by using the inverse variances of the data as weights.
w = 1./var(data);
[wcoeff,score,latent,tsquared,explained] = pca(data, 'VariableWeights',w);
% The first output, wcoeff, contains the coefficients of the principal components.

% The first three principal component coefficient vectors are:
c3 = wcoeff(:,1:3);

% Transform the coefficients so that they are orthonormal.
coefforth = diag(sqrt(w))*wcoeff;

% Create a plot of the first three columns of score.
figure()
hold on;
% plot(score(1:4844,1),score(1:4844,2),'x')
% plot(score(4845:end,1),score(4845:end,2),'o')
scatter3(score(1:608,1),score(1:608,2),score(1:608,3),'x');
scatter3(score(1213:end,1),score(1213:end,2),score(1213:end,3),'w');
scatter3(score(608:1213,1),score(608:1213,2),score(608:1213,3),'o');

xlabel('1st Principal Component')
ylabel('2nd Principal Component')
zlabel('3rd Principal Component')

%% Sparsify and export

M = sparse(documents);
save('/Users/Catalin/Desktop/documents.mat', 'M');