function [x1,y1,times,history] = nqL1rpvH(A,b,gamma,w,opts)

% Solve nonconvex quadratic L1 regularized problem via BSALPSQP with
% variable Hessian and performing the Armijo line search
%
% [x,y,history] = nqL1rpvH(gamma,tol,beta,para)
%
% Solves the following problem via BSALPSQP:
%
%  min 2*gamma ||x||_1+0.5||Ay-b||^2-gamma ||y||^2 s.t. x-y=0£¬-e¡Üx¡Üe
%
% The solution is returned in the vector (x,y).
%
% history is a structure that contains the objective value, the primal and
% dual residual norms, and the tolerances for the primal and dual residual
% norms at each iteration.
%
% beta is the augmented Lagrangian parameter.
% gamma is the model parameter.
% rho and sigma are line search parameter.
% eta is the parameter controlling the positive definiteness of -Q
% psi(x)=L/2||x||^2-beta/2||Ax+By^k-b-lambda^k/beta||^2 is the distance generating function
%
% More information can be found in the paper linked at:
% http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
%
%
% parameter
[m,n] = size(A); % the size of matrix A.
rho = opts.rho; % line search parameter
sigma = opts.sigma; % compression ratio for line search
tol = opts.tol; 
beta = opts.beta;
s = opts.s; % the dual steplength
MAX_ITER = opts.MAX_ITER;

% eta = 10^-3; % accuracy controlling the positive definiteness of Hk

% t_start = tic;
% start the clock
t0 = cputime;

%% Global constants and defaults

QUIET    = 0;
% ABSTOL   = 1e-4;
% RELTOL   = 1e-2;

% save a matrix-matrix and matrix-vector multiplication
AtA = A'*A;
Atb = A'*b;



%% BSALPSQP solver
%% Initialization
x0 = zeros(n,1);
y0 = zeros(n,1);
lambda0 = zeros(n,1); % lambda0 = ones(m,1);
Hk0 = AtA+(beta-2*gamma)*eye(n);
L = chol(Hk0,'lower');
Hk0_1 = L'\(L\eye(n));
% Hk0_1 = invHk0(L);
% Hk0_1 = Hk0\eye(n);
% Hk0_1 = inv(Hk0);

if ~QUIET
    fprintf('%3s\t%s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
        'NL', 'pri norm', 'dua norm',  'objective', 'MSE');
end

for k = 1:MAX_ITER

    % x-update
    pk = y0-lambda0/beta;
    x1 = sign(pk).*max(min(abs(pk)-2*gamma/beta,1),0);

    % y-update 
    nablayL = AtA*y0-Atb-2*gamma*y0-lambda0-beta*(x1-y0);
    haty1 = y0-Hk0_1*nablayL;
    dyk = haty1-y0;
    t = 1;
    NL = 1; % count line search execution times
    y1_try = y0+t*dyk; % A,b,x,y,gamma
    Lbeta = objective(A,b,x1,y0,gamma)+lambda0'*(x1-y0)+beta/2*sum((x1-y0).^2);
    Lbetanew = objective(A,b,x1,y1_try,gamma)+lambda0'*(x1-y1_try)+beta/2*sum((x1-y1_try).^2); 
    while Lbetanew > Lbeta-rho*t*dyk'*Hk0*dyk && t>10^-10 
        t = t*sigma;
        NL = NL+1;
        y1_try = y0+t*dyk;
        Lbetanew = objective(A,b,x1,y1_try,gamma)+lambda0'*(x1-y1_try)+beta/2*sum((x1-y1_try).^2);
    end
    y1 = y1_try;
        
    % lambda-update
    lambda1 = lambda0-s*(x1-y1);
    
    if max(max(norm(x1-x0,inf),norm(dyk,inf)),norm(lambda1-lambda0,inf))< tol
%     if (norm(x1-x0)+norm(y1-y0)+norm(lambda1-lambda0))/(norm(x1)+norm(y1)+norm(lambda1)+1)<tol
%         a1 = norm(x1-x0,inf)
%         a2 = norm(dyk,inf)
%         a3 = norm(s*(A*x0+B*y0-b),inf)
        disp('successful,congratulations!');
        history.objval(k)  = objective(A,b,x1,y1,gamma);
        history.pri_norm(k)  = norm(x1 - x0)+norm(y1 - y0);
        history.d_norm(k)  = norm(lambda1-lambda0);
%         history.mse(k) = sum((x1-w).^2)/n;
        history.mse(k) = sum((y1-w).^2)/n;
        times(k) = cputime-t0;
        if ~QUIET
            fprintf('%3d\t%10.4f\t%10.4f\t%10.2f\t%10.4f\n', k, ...
            history.pri_norm(k), history.d_norm(k), ...
            history.objval(k), history.mse(k));
        end
        break
    end
    
    % matrices-update
%     if eigQ>eta
%         hk0 = Q;
%     elseif abs(eigQ)<=eta
%         hk0 = Q+(eta-eigQ)*eye(n2);
%     else
%         hk0 = Q+2*abs(eigQ)*eye(n2);
%     end
%     Hk0 = hk0+beta*BtB;
    
    % diagnostics, reporting, termination checks
    history.objval(k)  = objective(A,b,x1,y1,gamma);
    history.pri_norm(k)  = norm(x1 - x0)+norm(y1 - y0);
%     history.y_norm(k)  = norm(y1 - y0);
    history.d_norm(k)  = norm(lambda1-lambda0);
%     history.mse(k) = sum((x1-w).^2)/n;
    history.mse(k) = sum((y1-w).^2)/n;
    
    if ~QUIET
        fprintf('%3d\t%3d\t%10.4f\t%10.4f\t%10.2f\t%10.4f\n', k, ...
            NL, history.pri_norm(k), history.d_norm(k), ...
            history.objval(k), history.mse(k));
    end
    
    % iterate-update
    x0 = x1;
    y0 = y1;
    lambda0 = lambda1;
    times(k) = cputime-t0;
end
%  Tcpu= toc(t_start);
end

function p = objective(A,b,x,y,gamma)
    p = ( 2*gamma*norm(x,1)+0.5*sum((A*y-b).^2)-gamma*y'*y );
end

% function H = invHk0(L)   % time-consuming
%     [m,n] = size(L);
%     if m~=n
%         error('Error using the programming. Matrix must be square!');
%     end
%     H = speye(n);
%     for i=1:n
%         H(i,1:i) = H(i,1:i)/L(i,i);
%         for j=i+1:n
%             H(j,1:i)=H(j,1:i)-H(i,1:i).*L(j,1:i);
%         end
%     end
% end