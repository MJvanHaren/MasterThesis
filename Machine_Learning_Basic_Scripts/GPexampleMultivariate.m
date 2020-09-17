close all; clear all;clc;
% rng(randperm(100,1),'twister')
rng('shuffle');
% trueF = @(x,z) 3*sin(0.9*x)+cos(0.5*z);
% trueF = @(x,z) 1*x.^2 + 0.5*z.^2;
% trueF = @(x,z) 1*x.^2 + 0.12*z.^3;
trueF = @(x,z) 0.1*x.^2 + sin(0.6*z);
h = @(x,z) [ones(length(x),1) x x.^2 z z.^2];
mf=0;

n = 50;                                     % number of test points
N = 4;                                     % number of training points (sqrt)

s = 0.05;                                % noise variance on data
dist = 10;
% Training data x &  y
% x1 = ((rand(N,1)-0.5)*dist);
% x2 = ((rand(N,1)-0.5)*dist);
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
%% mean function least squares
H = h(x(:,1),x(:,2))';
betaBar = inv(H*H')*H*y;
%% optimization of hyper parameters
options = optimoptions('fmincon','Display','iter',...
    'Algorithm','interior-point',...          % interior point does not work correctly with specifyobjectivegradient on
    'SpecifyObjectiveGradient',false,...
    'CheckGradients',false,...
    'StepTolerance',1e-10);
xres0 = [0.5 0.5 1e-2 1];
lb = [1e-3 1e-3 1e-6 1e-3];
ub = [1e3 1e3 1e0 1e3];
if mf
    [xres,~] = fmincon(@(ohyp) marLikelihood3Hyp2DMeanFunc(x,y,ohyp,betaBar,h),xres0,[],[],[],[],lb,ub,[],options);
else
    [xres,~] = fmincon(@(ohyp) marLikelihood3Hyp2D(x,y,ohyp),xres0,[],[],[],[],lb,ub,[],options);
end

%% kernel function to our training data
k = GPSEKernel2D(x,x,xres(1:2));
Ky = xres(4)*k+xres(3)*eye(N^2);
L = chol(Ky,'lower');                                       % cholesky of kernel matrix


% cov for the test points

k_s = xres(4)*GPSEKernel2D(x,xtest,xres(1:2));

Lk = L \ k_s;

Ht = h(xtest(:,1),xtest(:,2))';
R = Ht-H*inv(Ky)*k_s;

if mf
    mu = (Lk') * (L \ y) + R'*betaBar;

else
    mu = (Lk') * (L \ y);
end

for i = 1:n
   MU(:,i) = mu(1+(i-1)*n:i*n,1); 
end

% kernel star star
if mf
    k_ss = xres(4)*GPSEKernel2D(xtest,xtest,xres(1:2))+ R'*inv(H*inv(Ky)*H')*R;
else
    k_ss = xres(4)*GPSEKernel2D(xtest,xtest,xres(1:2));
end
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


