clear all
clc
%set parameters
randn('seed',0); % 1 for the experiments in the paper
rand('seed',0); % 1 for the experiments in the paper

% The image went through a Gaussian blur of size 9¡Á9
% and standard deviation 4 (applied by the MATLAB functions imfilter and fspecial) followed
% by an additive zero-mean white Gaussian noise with standard deviation
% 10-3.
% f = double(imread('Camera.tif'))/255;
% H = fspecial('gaussian',[9 9],4)
% Ig = imfilter(f,G,'same');
% I = Ig + 0.001*randn(size(f)); 

f = rand(6)
[m n] = size(f);
middle = n/2 + 1;
h = zeros(m,n);
for i=-1:1
   for j=-1:1
      h(i+middle,j+middle)= (1/(1+i*i+j*j));
   end
end
h
% % center and normalize the blur
h = fftshift(h)   
h = h/sum(h(:))
% definde the function handles that compute 
% the blur and the conjugate blur.
R = @(x) real(ifft2(fft2(h).*fft2(x)));
H = R(f)
H_1 = inv(H)
% RT = @(x) real(ifft2(conj(fft2(h)).*fft2(x)));
% 
% % define the function handles that compute 
% % the products by W (inverse DWT) and W' (DWT)
% wav = daubcqf(2); % Daubechies filter coefficients
% W = @(x) midwt(x,wav,3); % Inverse discrete orthogonal wavelet transform
% WT = @(x) mdwt(x,wav,3); % Discrete orthogonal wavelet transform using the Mallat algorithm (1D and 2D)
% 
% %Finally define the function handles that compute 
% % the products by A = RW  and A' =W'*R' 
% A = @(x) R(W(x));
% AT = @(x) WT(RT(x));


% t = zeros(1,100);
% for n = 1:100
% A = rand(n,n);
% b = rand(n,1);
% tic;
% x = A\b;
% t(n) = toc;
% end
% plot(t)

% w = zeros(n,1);
% I = randperm(n);
% w(I(1:n_spikes)) = randn(n_spikes,1); % the original sparse signal
% b = A*w + nf*randn(m,1);
%         
% % parameter
% gamma =min(0.1,0.01*max(abs(A'*b)));   % 0.005*max(abs(R'*y));