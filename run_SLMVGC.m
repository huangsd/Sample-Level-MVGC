clear;
clc;
close all;

load("HW2.mat")

% For HW2, the preferable parameter setting is: alpha=50  beta = 0.005
% lambda = 0.5  gamma = 0.05
% For ORL_mtv, the preferable parameter setting is: alpha = 100  beta =
% 0.01  lambda = 1  gamma = 10


parfor i = 1:10
    % result_loop(i,:) = SLMVGC(data, labels, alpha, beta, lambda, gamma);   %acc, nmi, Pu, Fscore, Precision, Recall, ARI
    result_loop(i,:) = SLMVGC(data, labels, 50, 0.005, 0.5, 0.05);
end
    mean_acc= mean(result_loop(:,1));
    mean_nmi = mean(result_loop(:,2));
    mean_pu = mean(result_loop(:,3));
    mean_fscore = mean(result_loop(:,4));
    std_acc = std(result_loop(:,1));
    std_nmi = std(result_loop(:,2));
    std_pu = std(result_loop(:,3));
    std_fscore = std(result_loop(:,4));
    
    RES = [mean_nmi, mean_acc, mean_pu, mean_fscore, std_nmi, std_acc, std_pu, std_fscore];
    
clear data labels

