clc
clear all
%set parameters
randn('seed',0); % 1 for the experiments in the paper
rand('seed',0); % 1 for the experiments in the paper
% L = [1 0 0;-1 2 0;1 1 1];
% [m,n] = size(L);
% if m~=n
%     error('Error using the programming. Matrix must be square!');
% end
% % A = L*L'
% % A_1 = inv(A)
% % L_1 = inv(L)
% invL = speye(n);
% for i=1:n
%     invL(i,1:i) = invL(i,1:i)/L(i,i);
%     for j=i+1:n
%         invL(j,1:i)=invL(j,1:i)-invL(i,1:i).*L(j,1:i);
%     end
% end

m=2160;n=7680;
A = randn(m,n);
for j = 1:n
    A(:,j) = A(:,j)/norm(A(:,j)); % normalize A
end
H = A'*A+3*eye(n);
% [a,b]=size(H)
% opts.SYM = true;
% opts.POSDEF = true;
% B = eye(n);
% invH = linsolve(A,B,opts);
L = chol(H,'lower');
L = sparse(L);
Hk0_1 = L'\(L\eye(n));
b = H*ones(n,1)