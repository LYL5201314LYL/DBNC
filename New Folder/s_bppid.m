function [sys,x0,str,ts,simStateCompliance] = exppidf(t,x,u,flag)

switch flag,
    case 0,
        [sys,x0,str,ts,simStateCompliance]=mdlInitializeSizes;
    case 1,
        sys=mdlDerivatives(t,x,u);
    case 2,
        sys=mdlUpdate(t,x,u);
    case 3,
        sys=mdlOutputs(t,x,u);
    case 4,
        sys=mdlGetTimeOfNextVarHit(t,x,u);
    case 9,
        sys=mdlTerminate(t,x,u);
    otherwise
        DAStudio.error('Simulink:blocks:unhandledFlag', num2str(flag));
        
end

function [sys,x0,str,ts,simStateCompliance]=mdlInitializeSizes
sizes=simsizes;
sizes.NumContStates=0;
sizes.NumDiscStates=3;
sizes.NumOutputs=4;
sizes.NumInputs=7;
sizes.DirFeedthrough=1;
sizes.NumSampleTimes=1;
sys=simsizes(sizes);
x0=[1,0.1,10];
str=[];
ts=[0 0];
simStateCompliance = 'UnknownSimState';

function sys=mdlDerivatives(t,x,u)

sys = [x];
function sys=mdlUpdate(t,x,u)
T=0.1;
x=[u(5);x(2)+u(5)*T;(u(5)-u(4))/T];%e(k),e(k)求和,e(k)求微分
sys=[x(1);x(2);x(3)];


function sys=mdlOutputs(t,x,u)
xite=0.2;
alfa=0.05;
IN=3;H=5;OUT=3;
T=0.1;
global wi;
global wo;
global wi_1;
global wi_2;
global wi_3;

global wo_1;
global wo_2;
global wo_3;

global Kp;
global Ki;
global Kd;
%wi=rand(5,3);wo=rand(3,5);

Oh=zeros(1,5);
I=Oh;
K1=zeros(3,1);K=zeros(1,3);dyu=0;dK=zeros(1,3);delta3=zeros(1,3);
d_wo=zeros(3,5);segma=zeros(1,5);delta2=zeros(1,5);d_wi=zeros(5,3);dO=zeros(1,5);
xi=[u(1),u(3),u(5)];
epid=[x(1);x(2);x(3)];%用于传给pid控制器的误差
I=xi*wi';%隐含层的输入
for j=1:1:5
    Oh(j)=(exp(I(j))-exp(-I(j)))/(exp(I(j))+exp(-I(j)));%隐含层的输出
end
K1=wo*Oh';  %输出层的输入
for i=1:1:3
    K(i)=exp(K1(i))/(exp(K1(i))+exp(-K(i)));%输出层的输出
end
u_k=K*epid;
%以下是权值调整
%隐含层至输出层的权值调整
dyu=sign((u(3)-u(2))/(u(7)-u(6)+0.01));
for j=1:1:3
    dK(j)=2/(exp(K1(j))+exp(-K1(j)))^2; %输出层的输出的一阶导
end
for i=1:1:3
    delta3(i)=u(5)*dyu*epid(i)*dK(i);  %输出层的delta
end
for j=1:1:3
    for i=1:1:5
        d_wo=xite*delta3(j)*Oh(i)+alfa*(wo_1-wo_2);
    end
end
wo=wo_1+d_wo+alfa*(wo_1-wo_2);
%以下是输入层至隐含层的权值调整
for i=1:1:5
    dO(i)=4/(exp(I(i))+exp(-I(i)))^2;
end
segma=delta3*wo;
for i=1:1:5
    delta2(i)=dO(i)*segma(i);
end
%for i=1:1:5
   % for j=1:1:3
      %  d_wi=xite*delta2(i)*xi(j)+alfa*(wi_1-wi_2);
   % end
%end
d_wi=xite*delta2'*xi;
wi=wi_1+d_wi+alfa*(wi_1-wi_2);

wo_3=wo_2;
wo_2=wo_1;
wo_1=wo;
wi_3=wi_2;
wi_2=wi_1;
wi_1=wi;
Kp=K(1);Ki=K(2);Kd=K(3);
sys=[u_k,Kp,Ki,Kd];
%sys=[1,2,3,4];

function sys=mdlGetTimeOfNextVarHit(t,x,u)

sampleTime = 0.1;    %  Example, set the next hit to be one second later.
sys = t + sampleTime;

function sys=mdlTerminate(t,x,u)

sys = [];







