close all; clear all;clc;
% rng(randperm(100,1),'twister')
rng('default');
trueF = @(x,z) 3*sin(0.9*x)+cos(0.5*z);


n =250;                                     % number of test points
N = 10;                                     % number of training points (sqrt)

s = 0.0005;                                % noise variance on data
dist = 10;
% Training data x &  y
% x = sort((rand(N,1)-0.5)*dist);
% z = sort((rand(N,1)-0.5)*dist);
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
hyp= [5;s;1];
k = GPSEKernel2D(x,x,[hyp(1);hyp(1)]);
% figure;
% surf(x,z,k','LineStyle','none')
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
k_ss = hyp(3)*GPSEKernel(X1,X1,hyp(1));

surf(xt1,xt2,MU,'LineStyle','none')
v = L\k_s;
s2 = k_ss-v'*v;
stdv = sqrt(s2);
%% mu, true function, samples and stdv plot

figure(2);clf;
subplot(1,3,1)
plot(X1,y,'+','MarkerSize',15); hold on;
plot(X1,trueF(X1),'LineWidth',1.3);
plot(X1,mu,'--','LineWidth',1.3);
plot(X1,mu-3*stdv','black');
plot(X1,mu+3*stdv','black');
inBetween = [(mu+3*stdv')' fliplr((mu-3*stdv')')];
x2 = [X1', fliplr(X1')];
hue = fill(x2,inBetween, [7 7 7]/8,'LineStyle','none');
alpha(0.3);
xlabel('input, x','interpreter','Latex');
ylabel('output, f(x)','interpreter','Latex');
title('grid+fmincon+mean func');

subplot(1,3,2)
plot(X1,y,'+','MarkerSize',15); hold on;
plot(X1,trueF(X1),'LineWidth',1.3);
plot(X1,mu0,'--','LineWidth',1.3);
plot(X1,mu0-3*stdv0','black');
plot(X1,mu0+3*stdv0','black');
inBetween = [(mu0+3*stdv0')' fliplr((mu0-3*stdv0')')];
x2 = [X1', fliplr(X1')];
hue = fill(x2,inBetween, [7 7 7]/8,'LineStyle','none');
alpha(0.3);
xlabel('input, x','interpreter','Latex');
ylabel('output, f(x)','interpreter','Latex');
title('without mean function')

subplot(1,3,3)
plot(X1,y,'+','MarkerSize',15); hold on;
plot(X1,trueF(X1),'LineWidth',1.3);
plot(X1,mum,'--','LineWidth',1.3);
plot(X1,mum-3*stdvm','black');
plot(X1,mum+3*stdvm','black');
inBetween = [(mum+3*stdvm')' fliplr((mum-3*stdvm')')];
x2 = [X1', fliplr(X1')];
hue = fill(x2,inBetween, [7 7 7]/8,'LineStyle','none');
alpha(0.3);
title('GPML minimize w/o mean func')
xlabel('input, x','interpreter','Latex');
ylabel('output, f(x)','interpreter','Latex');
legend('Generated samples','True function','Mu of fitted posterior function','$\mu \pm 3 \sigma$','interpreter','Latex')

%% Prior calcs and plots
% Lss_prior = chol(k_ss+1e-7*eye(n),'lower');
% f_prior = Lss_prior*randn(n,2);
% figure(3)
% plot(Xtest,f_prior);
% title('Two samples from prior with SE kernel')
% xlabel('x');
% ylabel('f(x)');
%% posterior calcs and plots
% Lss_post = chol(k_ss + 1e-7*eye(n)-Lk'*Lk,'lower');
% f_post = mu+Lss_post*randn(n,2);
% figure(4);
% plot(Xtest,f_post);
% title('Two samples from posterior with SE kernel')
% xlabel('x');
% ylabel('f(x)');