clear all
clc
randn('seed', 0); % ʹ�ó���ڶ������е����Ժ������randn��������һ��һ����������ɾ��������
rand('seed',0);   % �趨��ʼ״̬��ʹ�ó���ÿ�����еĽ�����һ�����еļ�������ͬ��

% % problem size
m = 50;
n1 = m*2;
n2 = m+1;
% % parameter
% gamma = 10^-3; % model parameter
% tol = 10^-4; % accuracy control
% % line search parameters
% rho = 0.1;
% sigma = 0.5;
% % Generate problem data
% Q = randn(para.n2);
% Q = (Q+Q')/2;
v = rand(n2,1);
V = eye(n2)-2*v*v'/(v'*v);
sigma=zeros(n2,1);
for i=1:n2
    sigma(i)=cos(i*pi/(n2+1))+1;
end
Sigma = diag(sigma);
Q = V*Sigma*V';
Q = 10*(-(Q+Q')/2+eye(n2));
lam1 = max(eig(Q))
LAM2 = min(eig(Q))



