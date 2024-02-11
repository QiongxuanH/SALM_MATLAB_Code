% Solves the following problem via BSALPSQP, VBSALPSQP and ADMM-g:
%
%   min 2*gamma ||x||_1+0.5||Ay-b||^2-gamma ||y||^2 s.t. x-y=0，-e≤x≤e
%
clear all
clc
%set parameters
randn('seed',0); % for reproducibility
rand('seed',0); % for reproducibility

fid = fopen('mytext.txt','w');
% NI=[];
% T=[];
% F=[];
    
% m = 2048;        % m = 2048;        % m = 1024;       % m = 512;
% n = 6000;        % n = 6500;        % n = 5120;       % n = 2560;
% n_spikes = 200;  % n_spikes = 200;  % n_spikes = 160; % n_spikes = 80;
% case 1
m = 1024;
n = 5120;
n_spikes = 160;
% % case 2
% m = 1024;
% n = 5120;
% n_spikes = 160;
% case 3
% m = 2048;
% n = 6000;
% n_spikes = 200;
% % case 4
% m = 2048;
% n = 6500;
% n_spikes = 200;
% % case 5
% m = 2048;
% n = 7000;
% n_spikes = 200;

nf = sqrt(0.001);
A = randn(m,n);
%         A = A*spdiags(1./sqrt(sum(A.^2))',0,n,n);
for j = 1:n
    A(:,j) = A(:,j)/norm(A(:,j)); % normalize A
end
w = zeros(n,1);
I = randperm(n);
w(I(1:n_spikes)) = randn(n_spikes,1); % the original sparse signal
b = A*w + nf*randn(m,1);
        
% parameter
gamma =min(0.1,0.01*max(abs(A'*b)));   % 0.005*max(abs(R'*y));

fprintf(' *********************************************\n')
fprintf('gamma %2.1e, \n', gamma)
fprintf(' *********************************************\n')     
 
fprintf(' ********** Comparison starts **********\n') % this line is edited on Oct 20, 2017
        
% parameter
opts.MAX_ITER = 1000; 
opts.tol = 1e-3;  % terminate criterion
opts.rho = 0.1;
opts.sigma = 0.5;
opts.s = 1e-3;
opts.beta = 2;
% BSALPSQP
[x1,y1,times_1,history1] = nqL1rpvH(A,b,gamma,w,opts);
Tcpu1 = times_1(end);
iter1 = length(history1.objval);
Fval1 = history1.objval(end);
MSE1 = history1.mse(end);
fprintf(' BSALPSQP terminated: iter %d, time %0.2f, fval %.3f, MSE %3.2e\n', iter1, Tcpu1, Fval1, MSE1)
        
%         % VBSALPSQP with variable Hessian without performing the Armijo line search
%         [x2,y2,Tcpu2,history2] = nqL1rpvHwl(A,b,gamma,w,opts);
%         iter2 = length(history2.objval);
%         Fval2 = history2.objval(end);
%         MSE2 = history2.mse(end);
%         fprintf(' VBSALPSQP terminated: iter %d, time %0.2f, fval %.3f, MSE %3.2e\n', iter2, Tcpu2, Fval2, MSE2)
       
% parameter        
clear opts
opts.tol = 1e-3; 
opts.beta = 2;
opts.s = 1e-3; % the dual steplength
opts.MAX_ITER = 1000;
% VBSALPSQP with constant Hessian
[x3,y3,times_3,history3] = nqL1rpcHwl(A,b,gamma,w,opts);
Tcpu3 = times_3(end);
iter3 = length(history3.objval);
Fval3 = history3.objval(end);
MSE3 = history3.mse(end);
fprintf(' VBSALPSQP_c terminated: iter %d, time %0.2f, fval %.3f, MSE %3.2e\n', iter3, Tcpu3, Fval3, MSE3)
        
% proximal ADMM-g
clear opts
opts.MAX_ITER = 1000; 
opts.tol = 1e-3; % terminate criterion
% proximal ADMM-g
[x4,y4,times_4,history4] = ProximalADMMg(A,b,gamma,w,opts);
Tcpu4 = times_4(end);
iter4 = length(history4.objval);
Fval4 = history4.objval(end);
MSE4 = history4.mse(end);
fprintf(' ADMM-g terminated: iter %d, time %0.2f, fval %.3f, MSE %3.2e\n', iter4, Tcpu4, Fval4, MSE4)
        
% Classical ADMM solver
% ADMM parameters
% Classical ADMM
clear opts
opts.MAX_ITER = 1000; 
opts.tol = 1e-3; % Termination criterion
% opts.beta = 1.618;
% opts.rho=1;
% opts.alpha=0.01;
[x5, y5, times_5, history5] = ADMM(A,b,gamma,w,opts);
Tcpu5 = times_5(end);
iter5 = length(history5.objval);
Fval5= history5.objval(end);
MSE5 = history5.mse(end);
fprintf(' ADMM terminated: iter %d, time %0.2f, fval %.3f, MSE %3.2e\n', iter5, Tcpu5, Fval5, MSE5);

% progress_r = [progress_r; gamma iter1, Tcpu1, Fval1, MSE1, iter3, Tcpu3, Fval3, MSE3, iter4, Tcpu4, Fval4, MSE4]; 
fprintf(fid, ' %5.0f & %5.0f & %3.2e & %.1f & %.2f & %.3f & %3.2e & %.1f & %.2f & %.3f & %3.2e & %.1f & %.2f & %.3f & %3.2e & %.1f & %.2f & %.3f & %3.2e\n', ...
    m, n, gamma, iter1, Tcpu1, Fval1, MSE1, iter3, Tcpu3, Fval3, MSE3, iter4, Tcpu4, Fval4, MSE4, iter5, Tcpu5, Fval5, MSE5);
fclose(fid);
% Display and save results

K1 = length(history1.mse);
K3 = length(history3.mse);
K4 = length(history4.mse);
K5 = length(history5.mse);
figure(1)
plot(1:K1, history1.mse, '-*r', 'LineWidth', 2);
hold on
plot(1:K3, history3.mse, '-+k', 'LineWidth', 2);
plot(1:K4, history4.mse, '-.b', 'LineWidth', 2);
plot(1:K5, history5.mse, '-.g', 'LineWidth', 2);
legend('BSALPSQP','VBSALPSQP','ADMM-g','ADMM')
set(gca,'FontName','Times','FontSize',16)
ylabel('MSE'); xlabel('Itr');
title(sprintf('m=%d, n=%d, r=%d, gamma=%g',m,n,n_spikes,gamma))
hold off

figure(2)
plot(times_1,history1.mse,'-*r', 'LineWidth',2)
hold on
plot(times_3,history3.mse,'-+k', 'LineWidth',2)
plot(times_4,history4.mse, '-.b', 'LineWidth', 2);
plot(times_5, history5.mse, '-.g', 'LineWidth', 2);
legend('BSALPSQP','VBSALPSQP','ADMM-g','ADMM')
set(gca,'FontName','Times','FontSize',16)
xlabel('CPU time (seconds)')
ylabel('MSE')
title(sprintf('m=%d, n=%d, r=%d, gamma=%g',m,n,n_spikes,gamma))
hold off

% figure(3)
% plot(1:K1, history1.objval, 'r-*', 'LineWidth', 2);
% hold on
% plot(1:K3, history3.objval, 'k-+', 'LineWidth', 2);
% plot(1:K4, history4.objval, 'b-.', 'LineWidth', 2);
% legend('BSALPSQP','VBSALPSQP','ADMM-g')
% set(gca,'FontName','Times','FontSize',16)
% ylabel('F(x,y)'); xlabel('Itr');
% title(sprintf('m=%d, n=%d, r=%d, gamma=%g',m,n,n_spikes,gamma))
% hold off
% 
% figure(4)
% plot(times_1,history1.objval,'r-*','LineWidth',2)
% hold on
% plot(times_3,history3.objval,'k-+','LineWidth',2)
% plot(times_4,history4.objval, 'b-.', 'LineWidth', 2);
% legend('BSALPSQP','VBSALPSQP','ADMM-g')
% set(gca,'FontName','Times','FontSize',16)
% xlabel('CPU time (seconds)')
% ylabel('F(x,y)')
% title(sprintf('m=%d, n=%d, r=%d, gamma=%g',m,n,n_spikes,gamma))
% hold off
% 
% figure(5)
% scrsz = get(0,'ScreenSize');
% set(5,'Position',[10 scrsz(4)*0.1 0.9*scrsz(3)/2 3*scrsz(4)/4])
% subplot(4,1,1)
% plot(w,'LineWidth',1.1)
% top = max(w(:));
% bottom = min(w(:));
% v = [0 n+1 bottom-0.05*(top-bottom)  top+0.05*((top-bottom))];
% set(gca,'FontName','Times')
% set(gca,'FontSize',14)
% title(sprintf('Original (n = %g, number of nonzeros = %g)',n,n_spikes))
% axis(v)
% 
% subplot(4,1,2)
% plot(x1,'LineWidth',1.1)
% set(gca,'FontName','Times')
% set(gca,'FontSize',14)
% axis(v)
% title(sprintf('BSALPSQP (%g iterations, m = %g, gamma = %5.3g, MSE = %5.3g)',...
%     K1,m,gamma,MSE1))
% 
% subplot(4,1,3)
% plot(x3,'LineWidth',1.1)
% set(gca,'FontName','Times')
% set(gca,'FontSize',14)
% title(sprintf('VBSALPSQP (%g iterations, MSE = %0.4g)',...
%       K3,MSE3))
% axis(v)
% 
% subplot(4,1,4)
% plot(x4,'LineWidth',1.1)
% set(gca,'FontName','Times')
% set(gca,'FontSize',14)
% title(sprintf('ProximalADMMg (%g iterations, MSE = %0.4g)',...
%       K4,MSE4))
% axis(v)


%% %% plot performance file
% % clf;   %clf删除当前图形窗口中、
% %        %%句柄未被隐藏(即它们的HandleVisibility属性为on)的图形对象。
% % figure(1);
% % %subplot(2,2,1);
% % perf(NI,'logplot');
% % %title('迭代次数性能');
% % %set(gca,'ylim',[0.3,1]);
% % xlabel('\tau','Interpreter','tex');
% % ylabel('\rho(\tau)','Interpreter','tex');
% % legend('GIST' ,'ReNPG','IRL1e1','IRL1e2','IRL1e3','GIRL1e1');  %修改成这样，在最后加一个0，自动找位置，下面的figure同理
% % %subplot(2,2,2);
% % figure(2);
% % perf(T,'logplot');
% % %title('时间性能');
% % % set(gca,'ylim',[0.1,1]);
% % xlabel('\tau','Interpreter','tex');                     %% 给x轴加注
% % ylabel('\rho(\tau)','Interpreter','tex');               %% 给y轴加注
% % legend('GIST' ,'ReNPG','IRL1e1','IRL1e2','IRL1e3','GIRL1e1');%,'DY+','HZ','location','best');%% 线分类说明'JHS','DHS','VHS','JPRP','DPRP','JHS','DHS',
% % %subplot(2,2,3);
% % figure(3);
% % perf(F,'logplot');
% % %title('目标函数计算性能');
% % %set(gca,'ylim',[0.5,1]);
% % xlabel('\tau','Interpreter','tex');
% % ylabel('\rho(\tau)','Interpreter','tex');
% % legend('GIST'
% ,'ReNPG','IRL1e1','IRL1e2','IRL1e3','GIRL1e1');%,'DY+','HZ','location','best'); %'JFR','PRP',