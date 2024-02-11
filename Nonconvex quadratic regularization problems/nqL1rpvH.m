function [x1,y1,Tcpu,history] = nqL1rpvH(Q,A,B,b,lb,up,gamma,tol,beta,para)

% Solve nonconvex quadratic L1 regularized problem via BSALPSQP with
% variable Hessian and performing the Armijo line search
%
% [x,y,history] = nqL1rpvH(gamma,tol,beta,para)
%
% Solves the following problem via BSALPSQP:
%
%   min gamma |x|_1+0.5y^{T}Qy s.t. Ax+By=b£¬x¡Ý0£¬-e¡Üy¡Üe
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
m = para.m; % the number of rows of matrix A.
n1 = para.n1; % the dimension of x
n2 = para.n2; % the dimension of y
rho = para.rho; % line search parameter
sigma = para.sigma; % compression ratio for line search
s = para.s; % the dual steplength
L = para.L; % x-subproblem

eta = 10^-4; % accuracy controlling the positive definiteness of Hk

% case 1
eigQ = -1; % Calculate the minimum eigenvalue of Q

% % case 2
% eigQ = -10; % Calculate the minimum eigenvalue of Q

t_start = tic;

%% Global constants and defaults

QUIET    = 0;
MAX_ITER = 1000;
% ABSTOL   = 1e-4;
% RELTOL   = 1e-2;

% save a matrix-matrix multiplication
BtB = B'*B;


%% BSALPSQP solver
%% Initialization
x0 = zeros(n1,1);
y0 = zeros(n2,1);
lambda0 = zeros(m,1); % lambda0 = ones(m,1);
% matrices-update
if eigQ>eta
    hk0 = Q;
elseif abs(eigQ)<=eta
    hk0 = Q+(eta-eigQ)*eye(n2);
else
    hk0 = Q+2*abs(eigQ)*eye(n2);
end
Hk0 = hk0+beta*BtB;

if ~QUIET
    fprintf('%3s\t%s\t%10s\t%10s\t%10s\n', 'iter', ...
        'NL', 'pri norm', 'dua norm',  'objective');
end

for k = 1:MAX_ITER

    % x-update
    pk = x0-beta/L*A'*(A*x0+B*y0-b-lambda0/beta);
    x1 = max(pk-gamma/L,0);

    % y-update 
    nablayL = Q*y0-B'*lambda0+beta*B'*(A*x1+B*y0-b);
    F = nablayL-Hk0*y0;
%     haty1 = quadprog(Hk0,F,[],[],[],[],lb,up,y0);
    haty1 = cplexqp(Hk0,F,[],[],[],[],lb,up,y0);
    dyk = haty1-y0;
    t = 1;
    NL = 1; % count line search execution times
    y1_try = y0+t*dyk;
    Lbeta = objective(Q,x1,y0,gamma)-lambda0'*(A*x1+B*y0-b)+beta/2*sum((A*x1+B*y0-b).^2);
    Lbetanew = objective(Q,x1,y1_try,gamma)-lambda0'*(A*x1+B*y1_try-b)+beta/2*sum((A*x1+B*y1_try-b).^2); 
    while Lbetanew > Lbeta-rho*t*dyk'*Hk0*dyk && t>10^-10 
        t = t*sigma;
        NL = NL+1;
        y1_try = y0+t*dyk;
        Lbetanew = objective(Q,x1,y1_try,gamma)-lambda0'*(A*x1+B*y1_try-b)+beta/2*sum((A*x1+B*y1_try-b).^2);
    end
    y1 = y1_try;
        
    % lambda-update
    lambda1 = lambda0+s*(A*x1+B*y1-b);
    
    if max(max(norm(x1-x0,inf),norm(dyk,inf)),norm(lambda1-lambda0,inf))< tol
%     if (norm(x1-x0)+norm(y1-y0)+norm(lambda1-lambda0))/(norm(x1)+norm(y1)+norm(lambda1)+1)<tol
%         a1 = norm(x1-x0,inf)
%         a2 = norm(dyk,inf)
%         a3 = norm(s*(A*x0+B*y0-b),inf)
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
        fprintf('%3d\t%3d\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            NL, history.pri_norm(k), history.d_norm(k), ...
            history.objval(k));
    end
    
    % iterate-update
    x0 = x1;
    y0 = y1;
    lambda0 = lambda1;
end
 Tcpu= toc(t_start);
end

function p = objective(Q,x,y,gamma)
    p = ( gamma*norm(x,1)+1/2*y'*Q*y );
end