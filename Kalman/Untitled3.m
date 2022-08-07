clc;
clear;
%动态贝叶斯网络建模
intra = zeros(2);
intra(1,2) = 1;
inter = zeros(2);
inter(1,1) = 1;
n = 2;

X = 2; % size of hidden state
Y = 2; % size of observable state

ns = [X Y];
dnodes = [];
onodes = [2];
eclass1 = [1 2];
eclass2 = [3 2];
bnet = mk_dbn(intra, inter, ns, 'discrete', dnodes, 'eclass1', eclass1, 'eclass2', eclass2);

x0 = rand(X,1);
V0 = eye(X); % must be positive semi definite!
C0 = rand(Y,X);
R0 = eye(Y);
A0 = rand(X,X);
Q0 = eye(X);

bnet.CPD{1} = gaussian_CPD(bnet, 1, 'mean', x0, 'cov', V0, 'cov_prior_weight', 0);
bnet.CPD{2} = gaussian_CPD(bnet, 2, 'mean', zeros(Y,1), 'cov', R0, 'weights', C0, ...
			   'clamp_mean', 1, 'cov_prior_weight', 0);
bnet.CPD{3} = gaussian_CPD(bnet, 3, 'mean', zeros(X,1), 'cov', Q0, 'weights', A0, ...
			   'clamp_mean', 1, 'cov_prior_weight', 0)