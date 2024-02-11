function [x,y,times,history] = lasso_lsqr(A, b, gamma, w, opts)
% lasso_lsqr Solve lasso problem via ADMM
%
% [z, history] = lasso_lsqr(A, b, lambda, rho, alpha);
%
% Solves the following problem via ADMM:
%
%   minimize 1/2*|| Ax - b ||_2^2 + \lambda || x ||_1,
%
% where A is a sparse matrix. This uses LSQR for the x-update instead.
%
% The solution is returned in the vector x.
%
% history is a structure that contains the objective value, the primal and
% dual residual norms, and the tolerances for the primal and dual residual
% norms at each iteration.
%
% rho is the augmented Lagrangian parameter.
%
% alpha is the over-relaxation parameter (typical values for alpha are
% between 1.0 and 1.8).
%
%
% More information can be found in the paper linked at:
% http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
%

QUIET    = 0;
MAX_ITER = 1000;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;

[m, n] = size(A);

x = zeros(n,1);
y = zeros(n,1);
u = zeros(n,1);


if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
    'lsqr iters', 'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

for k = 1:MAX_ITER

    % x-update with lsqr; uses previous x to warm start
    [x, flag, relres, iters] = lsqr([A; sqrt(w).*speye(n)], ...
        [b; sqrt(w)*(y-u)], [], [], [], [], x);

    if(flag ~=0)
        error('LSQR problem...\n');
    end

    % z-update with relaxation
    zold = y;
    x_hat = opts*x + (1 - opts)*zold;
    y = shrinkage(x_hat + u, gamma/w);

    u = u + (x_hat - y);

    % diagnostics, reporting, termination checks
    history.objval(k)  = objective(A, b, gamma, x, y);

    history.lsqr_iters(k) = iters;
    history.r_norm(k)  = norm(x - y);
    history.s_norm(k)  = norm(-w*(y - zold));

    history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-y));
    history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(w*u);

    if ~QUIET
        fprintf('%3d\t%10d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            sum(history.lsqr_iters), history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end

    if (history.r_norm(k) < history.eps_pri(k) && ...
      history.s_norm(k) < history.eps_dual(k))
    times(k) = cputime-t0;
        break;
    end

end

if ~QUIET
     times(k) = cputime-t0;
end
end

function p = objective(A, b, lambda, x, z)
    p = ( 1/2*sum((A*x - b).^2) + lambda*norm(z,1) );
end

function z = shrinkage(x, kappa)
    z = max( 0, x - kappa ) - max( 0, -x - kappa );
end