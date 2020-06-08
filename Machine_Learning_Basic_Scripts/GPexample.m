close all; clear all;clc;
% rng(randperm(100,1),'twister')
rng('default');
trueF = @(x) 3*sin(0.9*x)+3+x+0.1*x.^2;


n =250;                                     % number of test points
N = 8;                                     % number of training points

s = 0.005;                                % noise variance on data
dist = 10;
% Training data x &  y
X = (rand(N,1)-0.5)*dist;
y = trueF(X) + s^2*randn(N,1);
Xtest = linspace(-dist/0.5,dist/0.5,n)';               % test points

% basis for mean functions
h = @(x) [ones(length(x),1) x x.^2 x.^3];
mh = size(h(1),2);
%% hyper parameter optimization
x01 = linspace(0.01,60,10);
x02 = logspace(-12,-2,11);
x03 = logspace(-2,2,10);
x04 = linspace(1,1000,20);
[X01,X02,X03,X04] = ndgrid(x01,x02,x03,x04);

x0 = [2;5e-3;1];
ub = [10000,1,10000,900];         % lower and upper bounds for hyper parameters
lb= [0.05^2 1e-14 0.05^2,-900];

options = optimoptions('fmincon','Display','off',...
    'Algorithm','interior-point',...          % interior point does not work correctly with specifyobjectivegradient on
    'SpecifyObjectiveGradient',false,...
    'CheckGradients',false,...
    'StepTolerance',1e-10);
for i = 1:(size(X01,1))
    for j = 1:(size(X02,2))
        for ii = 1:size(X03,3)
            for jj = 1:size(X04,4)
              fval(i,j,ii,jj) = marLikelihood4hyp(X,y,h,[X01(i,j,ii,jj); X02(i,j,ii,jj); X03(i,j,ii,jj); X04(i,j,ii,jj)]);
            end
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
elseif (length(size(X01)))==3
    mini = min(min(min(fval)));
    [I]=find(fval==mini);
else
    mini = min(min(min(min(fval))));
    [I]=find(fval==mini);
end

xres0 = [X01(I); X02(I);X03(I);X04(I)];
[xres,~] = fmincon(@(x) marLikelihood4hyp(X,y,h,x),xres0,[],[],[],[],lb,ub,[],options);
meanfunc = [];
covfunc = @covSEiso;                        % Squared Exponental covariance function
likfunc = @likGauss;                        % Gaussian likelihood
hypUpdate = struct('mean', [], 'cov', log([x0(1) x0(3)]), 'lik', log(x0(2)));
% hypUpdate = minimize(hyp,@gp, -1000,@infGaussLik, meanfunc, covfunc, likfunc, X, y);
% [mu2, var2] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, X, y, Xtest);
xresm = exp([hypUpdate.cov(1) hypUpdate.lik hypUpdate.cov(2)]);
%% kernel function to our training data
k = GPSEKernel(X,X,xres(1));
km = GPSEKernel(X,X,xresm(1)^2);

Ky = xres(3)*k+xres(2)*eye(N);
L = chol(Ky,'lower');                                       % cholesky of kernel matrix
Lm = chol(xresm(3)^2*km+xresm(2)^2*eye(N),'lower');         % cholesky of kernel matrix


% cov for the test points
k_s = xres(3)*GPSEKernel(X,Xtest,xres(1));
k_sm = xresm(3)^2*GPSEKernel(X,Xtest,xresm(1)^2);

Lk = L \ k_s;
Lkm = Lm \ k_sm;

% mean function shit
B = xres(4)*eye(mh);
H = h(X)';
betaBar = inv(H*inv(Ky)*H')*H*inv(Ky)*y;                        % small b = 0
% betaBar = inv(inv(B)+H*inv(Ky)*H')*(H*inv(Ky)*y+inv(B)*b);    % any small b
Hs = h(Xtest)';
R = Hs-H*inv(Ky)*k_s;

mu = (Lk') * (L \ y)+R'*betaBar;
mum = (Lkm') * (Lm \ y);
mu0 = (Lk') * (L \ y);


% kernel star star
k_ss0 = xres(3)*GPSEKernel(Xtest,Xtest,xres(1));
k_ss = k_ss0 + R'*inv(H*inv(Ky)*H')*R;                % small b
% k_ss = k_ss+R'*inv(inv(B)+H*inv(Ky)*H')*R;              % any b
k_ssm = xresm(3)^2*GPSEKernel(Xtest,Xtest,xresm(1)^2);   % kernel at test points


s2 = diag(k_ss)' - sum(Lk.^2,1);
s2m = diag(k_ssm)' - sum(Lkm.^2,1);
s20 = diag(k_ss0)' - sum(Lk.^2,1);
stdv = sqrt(s2);
stdvm = sqrt(s2m);
stdv0 = sqrt(s20);

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
title('grid+fmincon+mean func');

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
title('without mean function')

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