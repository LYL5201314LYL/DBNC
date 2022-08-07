clc;
clear;
%%%%%%%%%%%%%%%%%%%%%%%%%%%% 动态贝叶斯网络建模%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ss = 3;  %总节点
%% 组内
intra = zeros(ss);   %每个time slice 有3个节点
intra(2,3) = 1;     %节点2-->3有向边连接
%% 组间
inter = zeros(ss);
inter([1 2],2) = 1;  %前一时刻的1和2节点，与后一时刻的2节点有向边‘-->’相连
%% 节点属性 
U = 1; %size of input
X = 2; % size of hidden state
Y = 1; % size of observable state
node_sizes = [U X Y];%节点的大小
onodes = [ 3];  %observable node 编号
dnodes = [];     %离散节点
eclass1 = [1 2 3 ];
eclass2 = [1 4 3];
eclass = [eclass1 eclass2];

%% 建立DBN
bnet = mk_dbn(intra, inter, node_sizes, 'discrete', dnodes, 'eclass1', eclass1, 'eclass2', eclass2, 'observed', onodes);
%输入值给定
%input 
u0_mean = 0;  %输入值的均值
Cov_u0 = 10000000;  %输入值的方差（选取较大的值）
%state*（根据卡尔曼模型计算所得）
A = [ 0.8768  -0.0469; 0.02345  0.994];
B = [ 0.2345; 0.02996];
X_weight = [0.2345 0.8768 -0.0469; 0.02996 0.02345  0.994];
C = [0 1];
D = 0;
%对于状态节点选取方差较小
x0_mean = zeros(X,1);
Cov_x0 = 100*eye(X);   %第一时间片的x初始化
x1_mean = zeros(X,1);
Cov_x1 = 100*eye(X);
%output （选取方差较小）
Y_mean = 0;
Cov_y = 1000;
%建立高斯分布
bnet.CPD{1} = gaussian_CPD(bnet, 1, 'mean', u0_mean, 'cov', Cov_u0, 'cov_prior_weight', 0);
bnet.CPD{2} = gaussian_CPD(bnet, 2, 'mean', x0_mean, 'cov', Cov_x0, 'cov_prior_weight', 0);
bnet.CPD{3} = gaussian_CPD(bnet, 3, 'mean', Y_mean, 'cov', Cov_y, 'weights', C, 'clamp_mean', 1,'cov_prior_weight', 0);
bnet.CPD{4} = gaussian_CPD(bnet, 5, 'mean', x1_mean, 'cov', Cov_x1, 'weights', X_weight, 'clamp_mean', 1,'cov_prior_weight', 0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 推断
T = 20; % 固定长度序列
clear engine;
engine = jtree_dbn_inf_engine(bnet);
evidence = cell(ss,T-2);%元胞数组
%得到输入和输出的 evidences
evidence{1,T-2} = 0;%输入u 13
evidence{3,T-2} = 0;%输出y  13
evidence{1,T-3} = 0;%输入u 12
evidence{3,T-3} = 0;%输出y  12
u = 0;
X_t = [0;0]; %俩个状态x1，x2，初始状态为0
%% %%%%%%%%%%%%%%%%%%%%%%%%
t = T-2;%13
for i = 1:100
    ut = evidence{1,t};  %t=13时刻的input
    %卡尔曼线性方程
    X_t1 = A*X_t + B*ut;
    Y_t = C*X_t;
    Y_t1 =  C*X_t1 ;
    X_t = X_t1;
    evidence{3,t+1} = Y_t1;%14
    evidence{3,t+2} = 10;   %15目标值
    evidence{3, t+3} = 10;   %16目标值
    evidence{3, t+4} = 10;   %17目标值
    evidence{3, t+5} = 10;   %18目标值
    %DBN网络推理计算input
    [engine,ll] = enter_evidence(engine, evidence);  %输入证据,注意不是数字11
     % m = marginal_nodes(engine, 1, t+1);  %求解input的边缘分布
     m1 = marginal_nodes(engine, 1, t+1);  %求解input的边缘分布
     m2 = marginal_nodes(engine, 1, t+2);  %求解input的边缘分布
     m3 = marginal_nodes(engine, 1, t+3);  %求解input的边缘分布
     m4 = marginal_nodes(engine, 1, t+4);  %求解input的边缘分布
     m = (4*m1.mu+3*m2.mu+2*m3.mu+1*m4.mu)/(4+3+2+1);  %a weighted sum of 4 inputs
    %.mu均值
    evidence(:,1:t) = evidence(:,2:t+1);%计算的新值替换，向左移动一个单位
    evidence{1,t} = m;  
    y(i)=Y_t1;
    u(i)=ut;
   
    
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot(y,'r');
hold on
 plot(u);
grid on
legend('输出Y','输入U')

