clc;
clear;
%%%%%%%%%%%%%%%%%%%%%%%%%%%% ��̬��Ҷ˹���罨ģ%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ss = 3;  %�ܽڵ�
%% ����
intra = zeros(ss);   %ÿ��time slice ��3���ڵ�
intra(2,3) = 1;     %�ڵ�2-->3���������
%% ���
inter = zeros(ss);
inter([1 2],2) = 1;  %ǰһʱ�̵�1��2�ڵ㣬���һʱ�̵�2�ڵ�����ߡ�-->������
%% �ڵ����� 
U = 1; %size of input
X = 2; % size of hidden state
Y = 1; % size of observable state
node_sizes = [U X Y];%�ڵ�Ĵ�С
onodes = [1 3];  %observable node ���
dnodes = [];     %��ɢ�ڵ�
eclass1 = [1 2 3 ];
eclass2 = [1 4 3];
eclass = [eclass1 eclass2];
%% ����DBN
bnet = mk_dbn(intra, inter, node_sizes, 'discrete', dnodes, 'eclass1', eclass1, 'eclass2', eclass2, 'observed', onodes);
%����ֵ����
%input 
u0_mean = 0;  %����ֵ�ľ�ֵ
Cov_u0 = 10000000;  %����ֵ�ķ���
%state
A = [ 0.8768  -0.0469; 0.02345  0.994];
B = [ 0.2345; 0.02996];
X_weight = [0.2345 0.8768 -0.0469; 0.02996 0.02345  0.994];
C = [0 1];
D = 0;

x0_mean = zeros(X,1);
Cov_x0 = 100*eye(X);   %��һʱ��Ƭ��x��ʼ��
x1_mean = zeros(X,1);
Cov_x1 = 100*eye(X);
%output
Y_mean = 0;
Cov_y = 1000;
%������˹�ֲ�
bnet.CPD{1} = gaussian_CPD(bnet, 1, 'mean', u0_mean, 'cov', Cov_u0, 'cov_prior_weight', 0);
bnet.CPD{2} = gaussian_CPD(bnet, 2, 'mean', x0_mean, 'cov', Cov_x0, 'cov_prior_weight', 0);
bnet.CPD{3} = gaussian_CPD(bnet, 3, 'mean', Y_mean, 'cov', Cov_y, 'weights', C, 'clamp_mean', 1,'cov_prior_weight', 0);
bnet.CPD{4} = gaussian_CPD(bnet, 5, 'mean', x1_mean, 'cov', Cov_x1, 'weights', X_weight, 'clamp_mean', 1,'cov_prior_weight', 0);
%% ģ�����
T = 20; % �̶���������
clear engine;
engine = jtree_dbn_inf_engine(bnet);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% �ƶ�
evidence = cell(ss,T-2);%Ԫ������
%�õ����������� evidences
evidence{1,T-2} = 0;%����
evidence{3,T-2} = 0;%���
evidence{1,T-3} = 0;
evidence{3,T-3} = 0;
%ÿ��ѡȡ5��ʱ�̵�slice����һ�Σ�ÿ������u��Ϊevidence������u���䣬u=1������600��
u = 0;
X_t = [0;0]; %����״̬x1��x2����ʼ״̬Ϊ0

t = T-2;
for i = 1:150
    ut = evidence{1,t};  %tʱ�̵�input
    %���������Է���
    X_t1 = A*X_t + B*ut;
    %Y_t = C*X_t;
    Y_t1 =  C(1,1:2)*X_t1 ;
    X_t = X_t1;
    evidence{3,t+1} = Y_t1;
    evidence{3,t+2} = 10;   %desired value
    evidence{3, t+3} = 10;   %desired value
    evidence{3, t+4} = 10;   %desired value
    evidence{3, t+5} = 10;   %desired value
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
legend('���Y','����U')

