close all; clear all;clc;
% rng(randperm(100,1),'twister')
rng('shuffle');
trueF = @(x,z) 3*sin(0.9*x)+cos(0.5*z);


n = 50;                                     % number of test points
N = 10;                                     % number of training points (sqrt)

s = 0.005;                                % noise variance on data
dist = 10;
% Training data x &  y
% x1 = sort((rand(N,1)-0.5)*dist);
% x2 = sort((rand(N,1)-0.5)*dist);
x1 = linspace(-dist/2,dist/2,N);
x2 = linspace(-dist/2,dist/2,N);

[X1,X2] = meshgrid(x1,x2);
Y = trueF(X1,X2) + s*randn(N,1);
for i = 1:N
   y(1+(i-1)*N:i*N,1) = Y(:,i);
   x(1+(i-1)*N:i*N,1) = X1(:,i);
   x(1+(i-1)*N:i*N,2) = X2(:,i);
end

xt1 = linspace(-dist/2,dist/2,n)';               % test points
xt2 = linspace(-dist/2,dist/2,n)';               % test points
[Xt1,Xt2] = meshgrid(xt1,xt2);
xtest = zeros(n*n,2);
for i = 1:n
   xtest(1+(i-1)*n:i*n,1) = Xt1(:,i);
   xtest(1+(i-1)*n:i*n,2) = Xt2(:,i);
end


%% kernel function to our training data
hyp= [1;s;1];
k = GPSEKernel2D(x,x,[hyp(1);hyp(1)]);
Ky = hyp(3)*k+hyp(2)*eye(N^2);
L = chol(Ky,'lower');                                       % cholesky of kernel matrix


% cov for the test points

k_s = hyp(3)*GPSEKernel2D(x,xtest,[hyp(1);hyp(1)]);

Lk = L \ k_s;

mu = (Lk') * (L \ y);
for i = 1:n
   MU(:,i) = mu(1+(i-1)*n:i*n,1); 
end

% kernel star star
k_ss = hyp(3)*GPSEKernel2D(xtest,xtest,[hyp(1);hyp(1)]);
figure(1); clf;
subplot(1,2,1);
surf(xt1,xt2,MU,'LineStyle','none'); hold on;
plot3(x(:,1),x(:,2),y,'.','Color',[1 0 0],'MarkerSize',10);
xlabel('x1 direction');
ylabel('x2 direction');
zlabel('Estimated function value');
v = L\k_s;
s2array = diag(k_ss)' - sum(Lk.^2,1);
for i = 1:n
   s2(:,i) = s2array(1,1+(i-1)*n:i*n);
end
subplot(1,2,2)
surf(xt1,xt2,s2,'LineStyle','none');
xlabel('x1 direction');
ylabel('x2 direction');
zlabel('Estimated function values variance');
stdv = sqrt(s2);
