close all; clear all;

trueF = @(x) sin(0.9*x);

n = 50;                                     % number of test points
N = 8;                                     % number of training points
s = 0.00005;                                % noise variance on data
lenghtP = 3.3;

xexample = linspace(-5,5,n)';               % test points
k_ss = GPSEKernel(xexample,xexample,lenghtP);   % kernel at test points

% Training data x &  y
X = (rand(N,1)-0.5)*10;
% y = trueF(X) + s*randn(N,1);
y = trueF(X); % noiseless

% kernel function to our training data
k = GPSEKernel(X,X,lenghtP); 
L = chol(k+1e-15*eye(N),'lower');         % cholesky of kernel matrix

% mean for the test points
k_s = GPSEKernel(X,xexample,lenghtP);
Lk = L \ k_s;
mu = (Lk') * (L \ y)

% SD
 s2 = diag(k_ss)' - sum(Lk.^2,1);
 stdv = sqrt(s2);
 
 % draw samples from posterior at test points
Lss = chol(k_ss + 1e-9*eye(n)-Lk'*Lk,'lower');
Lss_prior = chol(k_ss+1e-15*eye(n));
f_post = mu+Lss*randn(n,N);
f_prior = Lss_prior*randn(n,N);



figure(1)
plot(X,y,'r*'); hold on;
plot(xexample,trueF(xexample),'b');
plot(xexample,mu,'r--');
plot(xexample,mu-3*stdv','black:');
plot(xexample,mu+3*stdv','black:');
inBetween = [(mu+3*stdv')' fliplr((mu-3*stdv')')];
x2 = [xexample', fliplr(xexample')];
hue = fill(x2,inBetween, [0.17 0.17 0.17],'LineStyle','none');
alpha(0.3);
legend('Generated samples','True function','Mu of fitted posterior function','mu +/- 3 sigma')

