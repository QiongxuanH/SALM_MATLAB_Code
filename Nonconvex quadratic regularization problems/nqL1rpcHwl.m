function [x1,y1,Tcpu,history] = nqL1rpcHwl(Q,A,B,b,lb,up,gamma,tol,para)

% Solve nonconvex quadratic L1 regularized problem via BSALPSQP with
% variable Hessian and without performing the Armijo line search
%
% [x,y,history] = nqL1rpvHwl(gamma,tol,beta,para)
%
% Solves the following problem via BSALPSQP:
%
%   min gamma |x|_1-0.5y^{T}Qy s.t. Ax+By=b£¬x¡Ý0£¬-e¡Üy¡Üe
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
% eta is the parameter controlling the positive definiteness of Hk
% psi(x)=L/2||x||^2-beta/2||Ax+By^k-b-lambda^k/beta||^2 is the distance generating function
%
% More information can be found in the paper linked at:
% http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
%
%
% parameter
m = para.m; % the number of rows of matrix A.
n1 = para.n1; % the dimension of x
n2 = para.n2; % the dimension of y
s = para.s; % the dual steplength
L = para.L; % x-subproblem
%
% compute the Lipschitz constant Lg for g(y)
lamQ = 1; % compute the Lipschitz constant Lg for g(y)
% if n2 < 2000
%     tstart = tic;
%     lamQ = norm(Q);
%     teig = toc(tstart);
% else
%     clear opts
%     opts.issym = 1;     
%     tstart = tic;       
%     lamQ = eigs(Q,1,'LM',opts);
%     teig = toc(tstart);  
% end
% fprintf(' time for eigenvalues %g %6.5e\n', teig, lamQ)
%
% lamQ = max(eigs(Q));
%
t_start = tic;

%% Global constants and defaults

QUIET    = 0;
MAX_ITER = 1000;
% ABSTOL   = 1e-4;
% RELTOL   = 1e-2;

% save a matrix-matrix multiply
BtB = B'*B;

% compute L for BtB
if n2 < 2000
    tstart = tic;
    lamBtB = norm(BtB);
    teig = toc(tstart);
else
    clear opts
    opts.issym = 1;     
    tstart = tic;       
    lamBtB = eigs(BtB,1,'LM',opts);
    teig = toc(tstart);  
end
fprintf(' time for eigenvalues %g %6.5e\n', teig, lamBtB)
beta = 4;
% beta = 10;


%% BSALPSQP solver
%% Initialization
x0 = zeros(n1,1);
y0 = zeros(n2,1);
lambda0 = zeros(m,1); % lambda0 = ones(m,1);

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\n', 'iter', ...
        'pri norm', 'dua norm',  'objective');
end

for k = 1:MAX_ITER

    % x-update
    pk = x0-beta/L*A'*(A*x0+B*y0-b-lambda0/beta);
    x1 = max(pk-gamma/L,0);

    % y-update 
    nablayL = Q*y0-B'*lambda0+beta*B'*(A*x1+B*y0-b);
    qk = y0-nablayL/(lamQ+beta*lamBtB);
    y1 = min(max(lb,qk),up); 
        
    % lambda-update
    lambda1 = lambda0+s*(A*x1+B*y1-b);
    
%     a1 = norm(x1-x0,inf)
%     a2 = norm(y1-y0,inf)
%     a3 = norm(lambda1-lambda0,inf)
    if max(max(norm(x1-x0,inf),norm(y1-y0,inf)),norm(lambda1-lambda0,inf))< tol
%     if (norm(x1-x0)+norm(y1-y0)+norm(lambda1-lambda0))/(norm(x0)+norm(y0)+norm(lambda0)+1)<tol
        disp('successful,congratulations!');
        history.objval(k)  = objective(Q,x1,y1,gamma);
        history.pri_norm(k)  = norm(x1 - x0)+norm(y1 - y0);
        history.d_norm(k)  = norm(lambda1-lambda0);
        if ~QUIET
            fprintf('%3d\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.pri_norm(k), history.d_norm(k), ...
            history.objval(k));
        end
        break
    end
    
    % diagnostics, reporting, termination checks
    history.objval(k)  = objective(Q,x1,y1,gamma);
    history.pri_norm(k)  = norm(x1 - x0)+norm(y1 - y0);
%     history.y_norm(k)  = norm(y1 - y0);
    history.d_norm(k)  = norm(lambda1-lambda0);
    
    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.pri_norm(k), history.d_norm(k), ...
            history.objval(k));
    end
    
    % iterate-update
    x0 = x1;
    y0 = y1;
    lambda0 = lambda1;
end
Tcpu = toc(t_start);
end

function p = objective(Q,x,y,gamma)
    p = ( gamma*norm(x,1)+1/2*y'*Q*y );
end