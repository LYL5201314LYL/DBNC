tic
clc;
clear;

% Make a linear dynamical system

ss = 4;  %slice size
intra = zeros(ss);   %ÿ��time slice ��3���ڵ�
intra(2,4) = 1;     %�ڵ�2-->4���������
intra(3,4) = 1;     %�ڵ�3-->4���������
inter = zeros(ss);
inter([1 2],2) = 1;  %ǰһʱ�̵�1��2�ڵ㣬���һʱ�̵�2�ڵ�����ߡ�-->������
inter(3,3) = 1;

onodes = [1 3];  %observable node ���
dnodes = [];     %��ɢ�ڵ�
U = 1; %size of input
X = 2; % size of hidden state
Y = 1; % size of observable state
Z = 1; % size of disturbance state

node_sizes = [U X Y Z];%�ڵ�Ĵ�С
eclass1 = [1 2 3 4];
eclass2 = [1 5 6 4];
eclass = [eclass1 eclass2];
%����DBN
bnet = mk_dbn(intra, inter, node_sizes, 'discrete', dnodes, 'eclass1', eclass1, 'eclass2', eclass2, 'observed', onodes);

%input 
%u0_mean = rand();  %����ֵ�ľ�ֵ
u0_mean = 0;  %����ֵ�ľ�ֵ
%Cov_u0 = 1;  %����ֵ�ķ���
Cov_u0 = 10000000;  %����ֵ�ķ���

%state
A = [ 0.5182  -3.7735; 0.0377    0.8956];
B = [ 7.5469; 0.2088];
X_weight = [ 7.5469 0.5182  -3.7735 ; 0.2088 0.0377    0.8956  ];
C = [0 1 1];
D = 0;
x0_mean = zeros(X,1);
Cov_x0 = 100*eye(X);   %��һʱ��Ƭ��x��ʼ��

x1_mean = zeros(X,1);
Cov_x1 = 100*eye(X);

 
%output
Y_mean = 0;
%Cov_y = 1;
Cov_y = 1000;

Z0_mean = 0; %disturbance
Cov_z0 = 1;
Z_mean = 0; %disturbance
Cov_z = 1;

%������˹�ֲ�

bnet.CPD{1} = gaussian_CPD(bnet, 1, 'mean', u0_mean, 'cov', Cov_u0, 'cov_prior_weight', 0);
bnet.CPD{2} = gaussian_CPD(bnet, 2, 'mean', x0_mean, 'cov', Cov_x0, 'cov_prior_weight', 0);
bnet.CPD{3} = gaussian_CPD(bnet, 3, 'mean', Z0_mean, 'cov', Cov_z0, 'cov_prior_weight', 0);
bnet.CPD{4} = gaussian_CPD(bnet, 4, 'mean', Y_mean, 'cov', Cov_y, 'weights', C, 'clamp_mean', 1,'cov_prior_weight', 0);
bnet.CPD{5} = gaussian_CPD(bnet, 6, 'mean', x1_mean, 'cov', Cov_x1, 'weights', X_weight, 'clamp_mean', 1,'cov_prior_weight', 0);
bnet.CPD{6} = gaussian_CPD(bnet, 7, 'mean', Z_mean, 'cov', Cov_z, 'weights', 1, 'clamp_mean', 1,'cov_prior_weight', 0);

%T = 5; % fixed length sequences
T = 20; % fixed length sequences

clear engine;
%engine{1} = kalman_inf_engine(bnet);
%engine{2} = jtree_unrolled_dbn_inf_engine(bnet, T);
%engine{3} = jtree_dbn_inf_engine(bnet);
engine = jtree_dbn_inf_engine(bnet);
%N = length(engine);

% inference

%ev = sample_dbn(bnet, T-2);
evidence = cell(ss,T-2);
%get input and output evidences

%evidence(onodes,:) = ev(onodes, :);
evidence{1,T-2} = 0;
evidence{4,T-2} = 0;
evedence{3,T-2} = 0;
evidence{1,T-3} = 0;
evidence{4,T-3} = 0;
evedence{3,T-3} = 0;
%ÿ��ѡȡ5��ʱ�̵�slice����һ�Σ�ÿ������u��Ϊevidence������u���䣬u=1������600��
u = 0;
X_t = [0;0]; %��ʼ״̬Ϊ0

t = T-2;
for i = 1:150
    
    if i< 80
        evidence{3,t}=0;
        %evidence{3,t+2} = 0;   %disturbance value
        %evidence{3, t+3} = 0;   
        %evidence{3, t+4} = 0;   
        %evidence{3, t+5} = 0;   
    end
    if i>=80
         evidence{3,t}=1;
        %evidence{3,t+2} = 1;   %disturbance value
        %evidence{3, t+3} = 1;   
        %evidence{3, t+4} = 1;   
       % evidence{3, t+5} = 1; 
    end
    ut = evidence{1,t};  %tʱ�̵�input
    zt = evidence{3,t};
    X_t1 = A*X_t + B*ut;
    %Y_t = C*X_t;
    Y_t1 =  C(1,1:2)*X_t1 + zt;
    X_t = X_t1;
    %evidence{3,t+1} = Y_t1;
    evidence{4,t+1} = Y_t1;
    
    evidence{4,t+2} = 10;   %desired value
    evidence{4, t+3} = 10;   %desired value
    evidence{4, t+4} = 10;   %desired value
    evidence{4, t+5} = 10;   %desired value
    
   % if i>100
    %    evidence{3,t+2} = [5];   %desired value
    %end
    
    %evidence(onodes,1:T-1) = evidence(onodes,2:T);
    %evidence(3,T) = {[0;1]}; %disired value :�ٶ�0��λ��1                    
    
    %DBN�����������input
    [engine,ll] = enter_evidence(engine, evidence);  %����֤��
%     m = marginal_nodes(engine, 1, t+1);  %���input�ı�Ե�ֲ�
    m1 = marginal_nodes(engine, 1, t+1);  %���input�ı�Ե�ֲ�
    m2 = marginal_nodes(engine, 1, t+2);  %���input�ı�Ե�ֲ�
    m3 = marginal_nodes(engine, 1, t+3);  %���input�ı�Ե�ֲ�
    m4 = marginal_nodes(engine, 1, t+4);  %���input�ı�Ե�ֲ�
    m = (4*m1.mu+3*m2.mu+2*m3.mu+1*m4.mu)/(4+3+2+1);  %a weighted sum of 4 inputs
    
    evidence(:,1:t) = evidence(:,2:t+1);
    %evidence{2,t}  =  X_t1;
%     evidence{1,t} = m.mu; 
    evidence{1,t} = m; 
   
    result(i) = Y_t1;
   % result(i) = Y_t1;
    
    u_input(i) = ut;
    
    
    
    %[engine,ll] = enter_evidence(engine, evidence);  %����֤��
    %m = marginal_nodes(engine, 3, T);  %����Ե�ֲ�
    %result(t-5) = {m.mu};   %��ֵmu
    %evidence(3,T) = result(t-5);
    
end


plot(result,'r');
hold on
plot(u_input);
grid on
%t = 5;
%query = [1 3];
toc
%[engine,ll] = enter_evidence(engine, evidence);
%m = marginal_nodes(engine, 1, t);
%m.mu





