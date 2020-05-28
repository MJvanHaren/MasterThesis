close all; clear all;clc;
% rng(randperm(100,1),'twister')
rng('default');
trueF = @(x) sin(0.9*x);


n =500;                                     % number of test points
N = 10;                                     % number of training points

s = 0.005;                                % noise variance on data
dist = 10;
% Training data x &  y
X = (rand(N,1)-0.5)*dist;
y = trueF(X) + s^2*randn(N,1);
% y = trueF(X); % noiseless
Xtest = linspace(-dist/2,dist/2,n)';               % test points
%% hyper parameter optimization

p = 15;                      % period for per. kernel   
lengthP =1;                  % periodic kernel only!
x01 = linspace(0.1,7.5,50);
x02 = logspace(-6,0,50);
x03 = logspace(-1,1,50);
% x03 = 1;
[X01,X02,X03] = meshgrid(x01,x02,x03);

x0 = [2;5e-3;1];
ub = [100,1,100];         % lower and upper bounds for hyper parameters
lb= [0.05 1e-7 0.05];

options = optimoptions('fmincon','Display','off',...
    'Algorithm','trust-region-reflective',...          % interior point does not work correctly with specifyobjectivegradient on
    'SpecifyObjectiveGradient',true,...
    'CheckGradients',false,...
    'StepTolerance',1e-10);
for i = 1:(size(X01,1))
    for j = 1:(size(X02,2))
        for ii = 1:size(X03,3)
%             [xres{i,j,ii},fval(i,j,ii)] = fmincon(@(x) marLikelihood3hyp(X,y,x),[X01(i,j,ii); X02(i,j,ii); X03(i,j,ii)],[],[],[],[],lb,ub,[],options);
              fval(i,j,ii) = marLikelihood3hyp(X,y,[X01(i,j,ii); X02(i,j,ii); X03(i,j,ii)]);
        end
    end
end
%%
if (length(size(X01)))<3
    figure(1);clf;
    mini = (min(min(fval)));
    [I]=find(fval==mini);
    fval(fval >= mini+2*abs(mini)) = mini+2*abs(mini);
    surf(X01,X02,fval,'LineStyle','none');
    contourf(X01,X02,-fval);
    xlabel('$l$','interpreter','Latex');
    ylabel('$\sigma_n$','interpreter','Latex')
    set(gca,'yscale','log');
else
    mini = min(min(min(fval)));
    [I]=find(fval==mini);
end

xres0 = [X01(I); X02(I);X03(I)];
[xres,~] = fmincon(@(x) marLikelihood3hyp(X,y,x),xres0,[],[],[],[],lb,ub,[],options);
meanfunc = [];
covfunc = @covSEiso;                        % Squared Exponental covariance function
likfunc = @likGauss;                        % Gaussian likelihood
hyp = struct('mean', [], 'cov', log([x0(1) x0(3)]), 'lik', log(x0(2)));
hypUpdate = minimize(hyp,@gp, -1000,@infGaussLik, meanfunc, covfunc, likfunc, X, y);
[mu2, var2] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, X, y, Xtest);
xresm = exp([hypUpdate.cov(1) hypUpdate.lik hypUpdate.cov(2)]);
%% kernel function to our training data
k = GPSEKernel(X,X,xres(1));
k0 = GPSEKernel(X,X,xres0(1));
km = GPSEKernel(X,X,xresm(1));
kp = PeriodicKernel(X,X,p,lengthP);

L = chol(xres(3)^2*k+xres(2)^2*eye(N),'lower');         % cholesky of kernel matrix
Lm = chol(xresm(3)^2*km+xresm(2)^2*eye(N),'lower');         % cholesky of kernel matrix
L0 = chol(xres0(3)^2*k0+xres0(2)^2*eye(N),'lower');         % cholesky of kernel matrix
Lp = chol(kp+s^2*eye(N),'lower'); 



% mean for the test points
k_s = xres(3)^2*GPSEKernel(X,Xtest,xres(1));
k_sm = xresm(3)^2*GPSEKernel(X,Xtest,xresm(1));
k_s0 = xres0(3)^2*GPSEKernel(X,Xtest,xres0(1));
k_sp = PeriodicKernel(X,Xtest,p,lengthP);

Lk = L \ k_s;
Lkm = Lm \ k_sm;
Lk0 = L0 \ k_s0;
Lkp = Lp \ k_sp;

mu = (Lk') * (L \ y);
mum = (Lkm') * (Lm \ y);
mu0 = (Lk0') * (L0 \ y);
mup = (Lkp') * (Lp \ y);

% SD
k_ss = xres(3)^2*GPSEKernel(Xtest,Xtest,xres(1));   % kernel at test points
k_ssm = xresm(3)^2*GPSEKernel(Xtest,Xtest,xresm(1));   % kernel at test points
k_ss0 = xres0(3)^2*GPSEKernel(Xtest,Xtest,xres0(1));   % kernel at test points
k_ssp = PeriodicKernel(Xtest,Xtest,p,lengthP);

s2 = diag(k_ss)' - sum(Lk.^2,1);
s2m = diag(k_ssm)' - sum(Lkm.^2,1);
s20 = diag(k_ss0)' - sum(Lk0.^2,1);
s2p = diag(k_ssp)' - sum(Lkp.^2,1);
stdv = sqrt(s2);
stdvm = sqrt(s2m);
stdv0 = sqrt(s20);
stdv2 = sqrt(s2p);

%% mu, true function, samples and stdv plot
figure(2);clf;
subplot(1,3,1)
plot(X,y,'+','MarkerSize',15); hold on;
plot(Xtest,trueF(Xtest),'LineWidth',1.3);
plot(Xtest,mu,'--','LineWidth',1.3);
plot(Xtest,mu-3*stdv','black');
plot(Xtest,mu+3*stdv','black');
inBetween = [(mu+3*stdv')' fliplr((mu-3*stdv')')];
x2 = [Xtest', fliplr(Xtest')];
hue = fill(x2,inBetween, [7 7 7]/8,'LineStyle','none');
alpha(0.3);
xlabel('input, x','interpreter','Latex');
ylabel('output, f(x)','interpreter','Latex');
title('own opt. then fmincon');
% legend('Generated samples','True function','Mu of fitted posterior function','$\mu \pm 3 \sigma$','interpreter','Latex')

subplot(1,3,2)
plot(X,y,'+','MarkerSize',15); hold on;
plot(Xtest,trueF(Xtest),'LineWidth',1.3);
plot(Xtest,mu0,'--','LineWidth',1.3);
plot(Xtest,mu0-3*stdv0','black');
plot(Xtest,mu0+3*stdv0','black');
inBetween = [(mu0+3*stdv0')' fliplr((mu0-3*stdv0')')];
x2 = [Xtest', fliplr(Xtest')];
hue = fill(x2,inBetween, [7 7 7]/8,'LineStyle','none');
alpha(0.3);
xlabel('input, x','interpreter','Latex');
ylabel('output, f(x)','interpreter','Latex');
title('own opt.')
% legend('Generated samples','True function','Mu of fitted posterior function','$\mu \pm 3 \sigma$','interpreter','Latex')

subplot(1,3,3)
plot(X,y,'+','MarkerSize',15); hold on;
plot(Xtest,trueF(Xtest),'LineWidth',1.3);
plot(Xtest,mum,'--','LineWidth',1.3);
plot(Xtest,mum-3*stdvm','black');
plot(Xtest,mum+3*stdvm','black');
inBetween = [(mum+3*stdvm')' fliplr((mum-3*stdvm')')];
x2 = [Xtest', fliplr(Xtest')];
hue = fill(x2,inBetween, [7 7 7]/8,'LineStyle','none');
alpha(0.3);
title('GPML minimize')
xlabel('input, x','interpreter','Latex');
ylabel('output, f(x)','interpreter','Latex');
legend('Generated samples','True function','Mu of fitted posterior function','$\mu \pm 3 \sigma$','interpreter','Latex')

%%
figure(3);clf;
plot(X,y,'+','MarkerSize',15); hold on;
plot(Xtest,trueF(Xtest),'LineWidth',1.3);
plot(Xtest,mup,'--','LineWidth',1.3);
plot(Xtest,mup-3*stdv2','black');
plot(Xtest,mup+3*stdv2','black');
inBetween = [(mup+3*stdv2')' fliplr((mup-3*stdv2')')];
x2 = [Xtest', fliplr(Xtest')];
hue = fill(x2,inBetween, [7 7 7]/8,'LineStyle','none');
alpha(0.3);
xlabel('input, x','interpreter','Latex');
ylabel('output, f(x)','interpreter','Latex');
legend('Generated samples','True function','Mu of fitted posterior function','$\mu \pm 3 \sigma$','interpreter','Latex')

%% Prior calcs and plots
Lss_prior = chol(k_ss+1e-9*eye(n),'lower');
Lss_priorp = chol(k_ssp+1e-9*eye(n),'lower');
f_prior = Lss_prior*randn(n,2);
f_priorp = Lss_priorp*randn(n,2);
figure(4)
plot(Xtest,f_prior);
title('Two samples from prior with SE kernel')
xlabel('x');
ylabel('f(x)');
figure(5)
plot(Xtest,f_priorp);
title('Two samples from prior with periodic kernel')
xlabel('x');
ylabel('f(x)');
%% posterior calcs and plots
Lss_post = chol(k_ss + 1e-9*eye(n)-Lk'*Lk,'lower');
f_post = mu+Lss_post*randn(n,2);
figure(6);
plot(Xtest,f_post);
title('Two samples from posterior with SE kernel')
xlabel('x');
ylabel('f(x)');