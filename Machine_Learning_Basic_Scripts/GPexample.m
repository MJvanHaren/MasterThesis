close all; clear all;
rng('default');
trueF = @(x) sin(0.9*x);
% trueF = @(x) sin(x)-cos(x);
set(groot, 'DefaultFigureRenderer', 'painters');
n =150;                                     % number of test points
N = 10;                                     % number of training points
s = 0.00;                                % noise variance on data
p = 15;
lengthP =1;

dist = 10;

% Training data x &  y
X = (rand(N,1)-0.5)*dist;
y = trueF(X) + s*randn(N,1);
% y = trueF(X); % noiseless

% kernel function to our training data
k = GPSEKernel(X,X,lengthP);
kp = PeriodicKernel(X,X,p,lengthP);
L = chol(k+s*eye(N),'lower');         % cholesky of kernel matrix
Lp = chol(kp+s*eye(N),'lower'); 

Xtest = linspace(-dist/2,dist/2,n)';               % test points

% mean for the test points
k_s = GPSEKernel(X,Xtest,lengthP);
k_sp = PeriodicKernel(X,Xtest,p,lengthP);

Lk = L \ k_s;
Lkp = Lp \ k_sp;

mu = (Lk') * (L \ y);
mu = (Lkp') * (Lp \ y);

% SD
k_ss = GPSEKernel(Xtest,Xtest,lengthP);   % kernel at test points
k_ssp = PeriodicKernel(Xtest,Xtest,p,lengthP);
s2 = diag(k_ss)' - sum(Lk.^2,1);
s2p = diag(k_ssp)' - sum(Lkp.^2,1);
stdv = sqrt(s2);
stdv2 = sqrt(s2p);

%% mu, true function, samples and stdv plot
figure(1);clf;
inBetween = [(mu+3*stdv')' fliplr((mu-3*stdv')')];
x2 = [Xtest', fliplr(Xtest')];
hue = fill(x2,inBetween, [7 7 7]/8); hold on;
alpha(0.3);
sam = plot(X,y,'+','MarkerSize',15,'Color',[0, 0.4470, 0.7410]); hold on;
true = plot(Xtest,trueF(Xtest),'LineWidth',1.3,'Color',	[0.8500, 0.3250, 0.0980]);
mup = plot(Xtest,mu,'--','LineWidth',1.3,'Color',[0.9290, 0.6940, 0.1250]);


xlabel('input, x','interpreter','Latex');
ylabel('output, f(x)','interpreter','Latex');
legend([sam, true, mup, hue],{'Generated samples','True function','$\mu$ of fitted posterior function','$\mu \pm 3 \sigma$'},'interpreter','Latex')

figure(2)
inBetween = [(mu+3*stdv2')' fliplr((mu-3*stdv2')')];
x2 = [Xtest', fliplr(Xtest')];
hue = fill(x2,inBetween, [7 7 7]/8); hold on;
alpha(0.3);
sam=plot(X,y,'+','MarkerSize',15,'Color',[0, 0.4470, 0.7410]); hold on;
true=plot(Xtest,trueF(Xtest),'LineWidth',1.3,'Color',	[0.8500, 0.3250, 0.0980]);
mup=plot(Xtest,mu,'LineWidth',1.3,'Color',[0.9290, 0.6940, 0.1250]);
xlabel('input, x','interpreter','Latex');
ylabel('output, f(x)','interpreter','Latex');
legend([sam, true, mup, hue],{'Generated samples','True function','$\mu$ of fitted posterior function','$\mu \pm 3 \sigma$'},'interpreter','Latex')

%% Prior calcs and plots
Lss_prior = chol(k_ss+1e-9*eye(n),'lower');
Lss_priorp = chol(k_ssp+1e-9*eye(n),'lower');
f_prior = Lss_prior*randn(n,2);
f_priorp = Lss_priorp*randn(n,2);
figure(3)
plot(Xtest,f_prior);
title('Two samples from prior with SE kernel')
xlabel('x');
ylabel('f(x)');
figure(4)
plot(Xtest,f_priorp);
title('Two samples from prior with periodic kernel')
xlabel('x');
ylabel('f(x)');
%% posterior calcs and plots
Lss_post = chol(k_ss + 1e-6*eye(n)-Lk'*Lk,'lower');
f_post = mu+Lss_post*randn(n,N);