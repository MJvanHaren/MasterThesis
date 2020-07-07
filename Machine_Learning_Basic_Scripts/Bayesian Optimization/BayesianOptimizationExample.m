clear all; close all; clc;
rng('shuffle');
%% true fucntion
trueFunction  = @(x) x(:).^2 .* sin(4*pi*x(:)).^6; % x \in [0 1]
N = 20;
n = 500;
S = 0.025;
%% gridded search
xGrid = linspace(0,1,N)';
yGrid = trueFunction(xGrid)+S*randn(N,1);
[yGridMax,iGrid] = max(yGrid);
xGridMax = xGrid(iGrid);
%% random search
xRandom = rand(N,1);
yRandom = trueFunction(xRandom)+S*randn(N,1);
[yRandomMax, iRandom] = max(yRandom);
xRandomMax = xRandom(iRandom);
%% Bayesian optimization and Gaussian process surrogate
xBayesian = zeros(N,1);
xBayesian(1:3,1) = rand(3,1);
yBayesian = zeros(N,1);
yBayesian(1:3,1) = trueFunction(xBayesian(1:3,1))+S*randn(3,1);
% GP in BayesOpt
hyp = [1e-2 S 2];
xBayesianTest = rand(1,n);
k = GPSEKernel(xBayesian(1:3,1),xBayesian(1:3,1),hyp(1));
Ky = hyp(3)*k+hyp(2)*eye(3);
L = chol(Ky,'lower');

k_s = hyp(3)*GPSEKernel(xBayesian(1:3,1),xBayesianTest,hyp(1));
Lk = L \ k_s;
mu = (Lk') * (L \ yBayesian(1:3,1)); % surrogate so-far
k_ss = hyp(3)*GPSEKernel(xBayesianTest,xBayesianTest,hyp(1));
s2 = diag(k_ss)' - sum(Lk.^2,1);
stdv = sqrt(s2);
figure(1)
subplot(2,1,1)
plot(xBayesian(1:3,1),yBayesian(1:3,1),'o'); hold on;
plot(linspace(0,1,1000),trueFunction(linspace(0,1,1000)));
p1 = plot(xBayesianTest,mu,'.','Color',[0.9290, 0.6940, 0.1250]);
xlabel('x coordinate');
ylabel('y value');
legend('Samples of function','true function','GP fit so far','Autoupdate','off');


subplot(2,1,2)
xlabel('x coordinate');
ylabel('Probability of Improvement');


for i = 4:N
    xTest = rand(1,n); % step 1.1 (Xsamples)
    [PI, muSoFar] = ProbabilityImprovement(xBayesian(1:i-1,1),xTest,yBayesian(1:i-1,1),hyp);
    [~, index] = max(PI); 
    xBayesian(i) = xTest(index);
    yBayesian(i) = trueFunction(xBayesian(i,1))+S*randn;
    subplot(2,1,1)
    plot(xBayesian(i,1),yBayesian(i,1),'o','Color',[0, 0.4470, 0.7410]);
    delete(p1);
    p1 = plot(xTest,muSoFar,'.','Color',[0.9290, 0.6940, 0.1250]);
    subplot(2,1,2)
    [xTestSorted,xTestSortedOrder] = sort(xTest);
    plot(xTestSorted,PI(xTestSortedOrder,:)); hold on;
    plot(xBayesian(i),PI(index),'^');
    hold off;
end
[yBayesianMax,index] = max(yBayesian);
xBayesianMax = xBayesian(index);
%% plotting
figure(2); clf;
plot(linspace(0,1,5000),trueFunction(linspace(0,1,5000))); hold on;
plot(xGridMax, trueFunction(xGridMax),'o');
plot(xRandomMax, trueFunction(xRandomMax),'^');
plot(xBayesianMax,trueFunction(xBayesianMax),'s');
legend('True function','Gridded search optimum','Random search optimum','Bayesian optimization optimum','Location','best')