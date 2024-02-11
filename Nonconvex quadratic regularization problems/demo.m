% nonconvex quadratic L1 regularized problem
% min gamma |x|_1+0.5y^{T}Qy s.t. Ax+By=b，x≥0，-e≤y≤e

clear all
clc
randn('seed',0); % 使得程序第二次运行到此以后，下面的randn产生于上一次一样的随机生成矩阵或向量
rand('seed',0);   % 设定初始状态，使得程序每次运行的结果与第一次运行的计算结果相同。

fid = fopen('mytext.txt','w');
for index=33:2:39
    % problem size
    para.m = 100*index;
    para.n1 = para.m*2;
    para.n2 = para.m+1;
    % set parameter
    gamma = 10^-3; % model parameter gamma = 10^-3;
    tol = 0.0015;  %terminate criterion 10^-3
    beta = 1; % penalty parameter for ALF % beta = 1; good performance
    % line search parameters
    para.rho = 0.1;
    para.sigma = 0.5; % compression ratio
    para.s = 10^-3; % steplength for dual
    
    % Generate problem data
    
    % case 1
    v = rand(para.n2,1);
    V = eye(para.n2)-2*v*v'/(v'*v);
    sigma1 = zeros(para.n2,1);
    for i=1:para.n2
        sigma1(i)=cos(i*pi/(para.n2+1))+1;
    end
    Sigma = diag(sigma1);
    Q = V*Sigma*V'; % Q is positive definite and has prescribed eigenvalues between (0,2).
    Q = -(Q+Q')/2+eye(para.n2);

% %     case 2
%     v = rand(para.n2,1);
%     V = eye(para.n2)-2*v*v'/(v'*v);
%     sigma1 = zeros(para.n2,1);
%     for i=1:para.n2
%         sigma1(i)=cos(i*pi/(para.n2+1))+1;
%     end
%     Sigma = diag(sigma1);
%     Q = V*Sigma*V'; % Q is positive definite and has prescribed eigenvalues between (0,2).
%     Q = 10*(-(Q+Q')/2+eye(para.n2));

% %     case 3
%     Q = randn(para.n2);
%     Q = (Q+Q')/2;

%     if Q==Q'
%         disp('Q is a symmetric matrix');  
%     else
%         disp('Q is not a symmetric matrix');
%     end
%
    A = (rand(para.m,para.n1)-0.5)*4;
%     A = ceil(A); % randopara.m integer para.matrix whose each elepara.ment is an integer in (-2,2]
%     for j = 1:para.n1
%         A(:,j) = A(:,j)/norm(A(:,j)); % norpara.malize rows of A
%     end
    % save matrix-mtraix multiplication
    AtA = A'*A;
    % compute the L_0 for the distance generating function psi
    if para.n1 < 2000
        tstart = tic;
        lamAtA = norm(AtA);
        teig = toc(tstart);
    else
        clear opts
        opts.issym = 1;
        tstart = tic;
        lamAtA = eigs(AtA,1,'LM',opts);
        teig = toc(tstart);
    end
    fprintf(' time for eigenvalues %g %6.5e\n', teig, lamAtA)
    % lamAtA = max(eig(AtA));
    para.L = beta*lamAtA+0.001;  % L0>beta*||A'*A||
    % 
    B = (rand(para.m,para.n2)-0.5)*4;
%     B = ceil(B);
    % subsystem has a solution
    b = B*ones(para.n2,1);
    lb = -ones(para.n2,1); % lower bound for y
    up = ones(para.n2,1);  % upper bound for y 
    % run BSALPSQP
    [x1,y1,T1,history1] = nqL1rpvH(Q,A,B,b,lb,up,gamma,tol,beta,para);
    Itr1 = length(history1.objval);
    objini1 = history1.objval(1);
    objend1 = history1.objval(end);
    % run VBSALPSQP
    [x2,y2,T2,history2] = nqL1rpvHwl(Q,A,B,b,lb,up,gamma,tol,beta,para);
    Itr2 = length(history2.objval);
    objini2 = history2.objval(1);
    objend2 = history2.objval(end);
%     % run VBSALPSQP with constant Hessian
%     [x3,y3,T3,history3] = nqL1rpcHwl(Q,A,B,b,lb,up,gamma,tol,para);
    fprintf(fid,' %d & %d & %d & %3.2e & %d & %.2f & %.3f & %.3f & %d & %.2f & %.3f & %.3f\n', ...
        para.m, para.n1, para.n2, lamAtA, Itr1, T1, objini1, objend1, Itr2, T2, objini2, objend2);
end
fclose(fid);


% K = length(history.objval);
% 
% h = figure;
% plot(1:K, history.objval, 'k', 'markerSize', 10, 'LineWidth', 2);
% ylabel('f(x^k) + g(z^k)'); xlabel('iter (k)');
% 
% g = figure;
% subplot(2,1,1);
% semilogy(1:K, max(1e-8, history.r_norm), 'k', ...
%     1:K, history.eps_pri, 'k--',  'LineWidth', 2);
% ylabel('||r||_2');
% 
% subplot(2,1,2);
% separa.milogy(1:K, max(1e-8, history.s_norm), 'k', ...
%     1:K, history.eps_dual, 'k--', 'LineWidth', 2);
% ylabel('||s||_2'); xlabel('iter (k)');
