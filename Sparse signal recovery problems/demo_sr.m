% Solves the following problem via BSALPSQP, VBSALPSQP and ADMM-g:
%
%   min 2*gamma ||x||_1+0.5||Ay-b||^2-gamma ||y||^2 s.t. x-y=0，-e≤x≤e
%
clear all
clc
%set parameters
randn('seed',0); % 1 for the experiments in the paper
rand('seed',0); % 1 for the experiments in the paper

fid = fopen('mytext.txt','w');
% NI=[];
% T=[];
% F=[];
for index = 1:0.5:8.5
    
    m = 256*index;
    n = 768*index;
    k = 80*index;
    nf = 0.01;
    tol = 1e-3; % terminate criterion
     
    % generate problems
    progress_r = [];
    for repeats = 1:4
        A = randn(m,n);
        for j = 1:n
            A(:,j) = A(:,j)/norm(A(:,j)); % normalize A
        end
        w = zeros(n,1);
        I = randperm(n);
        w(I(1:k)) = randn(k,1); % the original sparse signal
        b = A*w + nf*randn(m,1);
        
        % parameter
        gamma =min(0.1,0.01*max(abs(A'*b)));   % 0.005*max(abs(R'*y));

        fprintf(' *********************************************\n')
        fprintf(' index %2.1f, gamma %2.1e, \n', index, gamma)
        fprintf(' *********************************************\n')
        
%         clear opts
        opts.MAX_ITER = 1000; 
        opts.tol = tol; % terminate criterion
        opts.rho = 0.1;
        opts.sigma = 0.5;
        opts.s = 0.001;
        opts.beta = 3;
        fprintf(' ********** Comparison starts **********\n') % this line is edited on Oct 20, 2017
        
        % BSALPSQP
        [x1,y1,Tcpu1,history1] = nqL1rpvH(A,b,gamma,w,opts);
        iter1 = length(history1.objval);
        Fval1 = history1.objval(end);
        MSE1 = history1.mse(end);
        fprintf(' BSALPSQP terminated: iter %d, time %0.2f, fval %.3f, MSE %3.2e\n', iter1, Tcpu1, Fval1, MSE1)
        
%         % VBSALPSQP
%         [x2,y2,Tcpu2,history2] = nqL1rpvHwl(A,b,gamma,w,opts);
%         iter2 = length(history2.objval);
%         Fval2 = history2.objval(end);
%         MSE2 = history2.mse(end);
%         fprintf(' VBSALPSQP terminated: iter %d, time %0.2f, fval %.3f, MSE %3.2e\n', iter2, Tcpu2, Fval2, MSE2)
       
        % VBSALPSQP with constant Hessian
        [x3,y3,Tcpu3,history3] = nqL1rpcHwl(A,b,gamma,w,opts);
        iter3 = length(history3.objval);
        Fval3 = history3.objval(end);
        MSE3 = history3.mse(end);
        fprintf(' VBSALPSQP_c terminated: iter %d, time %0.2f, fval %.3f, MSE %3.2e\n', iter3, Tcpu3, Fval3, MSE3)
        
        % proximal ADMM-g
        [x4,y4,Tcpu4,history4] = ProximalADMMg(A,b,gamma,w,opts);
        iter4 = length(history4.objval);
        Fval4 = history4.objval(end);
        MSE4 = history4.mse(end);
        fprintf(' ProximalADMMg terminated: iter %d, time %0.2f, fval %.3f, MSE %3.2e\n', iter4, Tcpu4, Fval4, MSE4)
        
% %         % reweighted NPG
% %         [x0,iter0,Fval_log0,time0] = ReNPG(A,b,epsilon,lambda, tol,maxiter,5);
% %         fprintf(' reweighted NPG_1 terminated: iter %d, time %5.1f, fval %6.5e\n', iter0, time0, Fval_log0)
% %         
% %         % RPGe1
% %         %         profile on
% %         [x1,iter1,time1] = IRL1e1(A,b,epsilon,lambda, L, tol,maxiter,200);
% %         Fval_log1 = 0.5*norm(A*x1-b)^2 + lambda*sum(log(abs(x1)/epsilon + 1));
% %         fprintf(' RPGe1 terminated: iter %d, time %5.1f, fval %6.5e\n', iter1, time1, Fval_log1)
% %         
% %         % RPGe2
% %         [x2,iter2,time2] = IRL1e2(A,b,epsilon,lambda, L, tol,maxiter,50);
% %         Fval_log2 = 0.5*norm(A*x2-b)^2 + lambda*sum(log(abs(x2)/epsilon + 1));
% %         fprintf(' RPGe2 terminated: iter %d, time %5.1f, fval %6.5e\n', iter2, time2, Fval_log2)
% %         %         profile off
% %         
% %         % RPGe3
% %         [x3,iter3,time3] = IRL1e3(A,b,epsilon,lambda, L, tol,maxiter,50);
% %         Fval_log3 = 0.5*norm(A*x3-b)^2 + lambda*sum(log(abs(x3)/epsilon + 1));
% %         fprintf(' RPGe3 terminated: iter %d, time %5.1f, fval %6.5e\n', iter3, time3, Fval_log3)
% %         
% %         % G-SRPGe
% %         [ x4,iter4,time4 ] = GIRL1e1( A,b,epsilon,lambda, L, tol,maxiter,200, lb, ub);
% %         Fval_log4 = 0.5*norm(A*x4-b)^2 + lambda*sum(log(abs(x4)/epsilon + 1));
% %         fprintf(' G-SRPGe3 terminated: iter %d, time %5.1f, fval %6.5e\n', iter4, time4, Fval_log4)

        progress_r = [progress_r; gamma iter1, Tcpu1, Fval1, MSE1, iter3, Tcpu3, Fval3, MSE3, iter4, Tcpu4, Fval4, MSE4]; 
%         progress_r = [progress_r; gamma iter1, Tcpu1, Fval1, MSE1, iter2, Tcpu2, Fval2, MSE2, iter3, Tcpu3, Fval3, MSE3];        
    end
    aver=mean(progress_r);
    fprintf(fid,' %5.0f & %5.0f & %3.2e & %.1f & %.2f & %.3f & %3.2e & %.1f & %.2f & %.3f & %3.2e & %.1f & %.2f & %.3f & %3.2e\n', ...
        m, n, aver(1),aver(2),aver(3),aver(4),aver(5),aver(6),aver(7),aver(8),aver(9),aver(10),aver(11),aver(12),aver(13));
%     fprintf(fid,' %5.0f & %5.0f & %2.1e & %5.1f & %6.2e & %6.3f & %3.2e & %5.1f & %6.2e & %6.3f & %3.2e & %5.1f & %6.2e & %6.3f & %3.2e\n', ...
%         m, n, aver(1),aver(2),aver(3),aver(4),aver(5),aver(6),aver(7),aver(8),aver(9),aver(10),aver(11),aver(12),aver(13));
%     NI=[NI;aver(:,2:7)];
%     T=[T;aver(:,8:13)];
%     F=[F;aver(:,14:19)];
end
fclose(fid);

% % %% 画图
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
% % legend('GIST' ,'ReNPG','IRL1e1','IRL1e2','IRL1e3','GIRL1e1');%,'DY+','HZ','location','best'); %'JFR','PRP',