function [x1,y1,times,history] = ADMM(A,b,gamma,w,opts)

[m,n] = size(A); % the size of matrix A.
tol = opts.tol; 
MAX_ITER = opts.MAX_ITER;
% rho = opts.rho; 
% alpha = opts.alpha; 

t0 = cputime;


QUIET    = 0;
% ABSTOL   = 1e-4;
% RELTOL   = 1e-2;

AtA = A'*A;
Atb = A'*b;
rho = 2;
% Cholesky 分解， $R$ 为上三角矩阵且 $R^\top R=A^\top A + (\rho-2\gamma) I_n$。
% 与原始问题同样的，由于罚因子在算法的迭代过程中未变化，事先缓存 Cholesky 分解可以加速迭代过程。
W = AtA + rho*eye(n) - 2*gamma*eye(n);
R = chol(W);
% lam = eig(AtA-2*gamma*eye(n));
% eig_g = min(lam); % Calculate the minimum eigenvalue of the Hessian of g
% L_g = max(abs(eig_g),max(lam)); % compute the Lipschitz constant Lg for g(y)


x0 = zeros(n,1);
y0 = zeros(n,1);
lambda0 = zeros(n,1); % lambda0 = ones(m,1);

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
        'pri norm', 'dua norm',  'objective', 'MSE');
end

for k = 1:MAX_ITER

    % x-update
% 	 x1 = (AtA + rho * eye(n)) \ (Atb + rho * (y0 - lambda0));
     v = y0 - (1/rho)*lambda0;
     x1 = sign(v).*max(min(abs(v)-2*gamma/rho,1),0);
	 
	 % y-update with relaxation
     % y0 = y1;
 %     x_hat = alpha*x1 + (1 - alpha)*y0;
%     y1 = (AtA + (rho * eye(n) - 2*gamma * eye(n))) \ (rho * x1 + Atb + lambda0);
     h = rho * x1 + Atb + lambda0;
     y1 = R \ (R' \ h);

	% lambda-update with relaxation
	
    lambda1 = lambda0 + rho*(x1 - y1);
	
	if max(max(norm(x1-x0,inf),norm(y1-y0,inf)),norm(lambda1-lambda0,inf))< tol
        disp('successful,congratulations!');
        history.objval(k)  = objective(A,b,x1,y1,gamma);
% 		history.r_norm(k)  = norm(x1 - y1);
%         history.s_norm(k)  = norm(-rho*(y1 - y0));
        history.pri_norm(k)  = norm(x1 - x0)+norm(y1 - y0);
        history.d_norm(k)  = norm(lambda1-lambda0);
%         history.pri_norm(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x1), norm(-y1));
%         history.d_norm(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*lambda1);
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
    
    % iterate-update
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