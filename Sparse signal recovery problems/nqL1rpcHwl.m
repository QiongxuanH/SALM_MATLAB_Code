function [x1,y1,times,history] = nqL1rpcHwl(A,b,gamma,w,opts)

% the constrained piecewise quadratic approximation model
%
% min 0.5*||Ax-b||^2+gamma(2|x|_1-||x||^2) s.t. -e¡Üx¡Üe
%
% Solve the above problem via BSALPSQP with
% variable Hessian and without performing the Armijo line search
%
% [x,y,history] = nqL1rpvHwl(Q,A,B,b,lb,up,gamma,tol,beta,para)
%
% Solves the following problem via BSALPSQP:
%
%   min 2*gamma ||x||_1+0.5||Ay-b||^2-gamma ||y||^2 s.t. x-y=0£¬-e¡Üx¡Üe
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
% psi(x)=0 is the distance generating function
%
% More information can be found in the paper linked at:
% http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
%
%
% parameter
[m,n] = size(A); % the size of matrix A.
tol = opts.tol; 
beta = opts.beta;
s = opts.s; % the dual steplength
MAX_ITER = opts.MAX_ITER;

% eta = 10^-3; % accuracy controlling the positive definiteness of g

% t_start = tic;
% start the clock
t0 = cputime;

%% Global constants and defaults

QUIET    = 0;
% ABSTOL   = 1e-4;
% RELTOL   = 1e-2;

% save a matrix-matrix multiplication
AtA = A'*A;
Atb = A'*b;
lam = eig(AtA-2*gamma*eye(n));
eig_g = min(lam); % Calculate the minimum eigenvalue of the Hessian of g
L_g = max(abs(eig_g),max(lam)); % compute the Lipschitz constant Lg for g(y)
% if eig_g > L_g/2
%     Hk0 = AtA+(beta-2*gamma)*eye(n);
% elseif abs(eig_g) <= L_g/2
%     Hk0 = AtA+(beta-eig_g+L_g/2-2*gamma)*eye(n);
% else
%     Hk0 = AtA+(beta-2*gamma+2*abs(eig_g))*eye(n);
% end

%% BSALPSQP solver
%% Initialization
x0 = zeros(n,1);
y0 = zeros(n,1);
lambda0 = zeros(n,1); % lambda0 = ones(m,1);
% Hk0_1 = Hk0\eye(n);

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
        'pri norm', 'dua norm',  'objective', 'MSE');
end

for k = 1:MAX_ITER

    % x-update
    pk = y0-lambda0/beta;
    x1 = sign(pk).*max(min(abs(pk)-2*gamma/beta,1),0);

    % y-update 
    nablayL = AtA*y0-Atb-2*gamma*y0-lambda0-beta*(x1-y0);
    y1 = y0-nablayL/(L_g+beta);
   
    % lambda-update
    lambda1 = lambda0-s*(x1-y1);
    
    if max(max(norm(x1-x0,inf),norm(y1-y0,inf)),norm(lambda1-lambda0,inf))< tol
%     if (norm(x1-x0)+norm(y1-y0)+norm(lambda1-lambda0))/(norm(x0)+norm(y0)+norm(lambda0)+1)<tol
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
    
    % diagnostics, reporting, termination checks
    history.objval(k)  = objective(A,b,x1,y1,gamma);
    history.pri_norm(k)  = norm(x1 - x0)+norm(y1 - y0);
%     history.y_norm(k)  = norm(y1 - y0);
    history.d_norm(k)  = norm(lambda1-lambda0);
%     history.mse(k) = sum((x1-w).^2)/n;
    history.mse(k) = sum((y1-w).^2)/n;
    
    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.2f\t%10.4f\n', k, ...
            history.pri_norm(k), history.d_norm(k), ...
            history.objval(k), history.mse(k));
    end
    
    % iterate-update and store times
    x0 = x1;
    y0 = y1;
    lambda0 = lambda1;
    times(k) = cputime-t0;
end
% Tcpu = toc(t_start);
end

function p = objective(A,b,x,y,gamma)
    p = ( 2*gamma*norm(x,1)+0.5*sum((A*y-b).^2)-gamma*y'*y );
end
