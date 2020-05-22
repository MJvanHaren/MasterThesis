function [model] = GPR()
%% Gaussian Process (GP) regression
% N(f|mu,Sigma)
% mu = mu(xs) + Ks'inv(K)(f-mu(x))
% Sigma = Kss - Ks'inv(K)Ks

%clear all; close all; 
rng('default');

%% generate data
L = 1;
xs = (-5:0.2:5)'; %test data
ns = length(xs);
keps = 1e-8;

%define mean and kernel functions
muFn = @(x) 0*x(:).^2;
Kfn = @(x,z) 1*exp(-sq_dist(x'/L,z'/L)/2);

%% plot sampled functions from the prior
figure; hold on
for i=1:3
  model = struct('mu', muFn(xs), 'Sigma',  Kfn(xs, xs) + 1e-15*eye(size(xs, 1)));
  fs = gauss_sample(model, 1);
  plot(xs, fs, 'k-', 'linewidth', 2)
end
title('samples from GP prior');

%% generate noise-less training data
Xtrain = [-4, -3, -2, -1, 1]';
ftrain = sin(Xtrain);

%% compute posterior predictive
K = Kfn(Xtrain, Xtrain); % K
Ks = Kfn(Xtrain, xs); %K_*
Kss = Kfn(xs, xs) + keps*eye(length(xs)); % K_** (keps is essential!)
Ki = inv(K); %O(Ntrain^3)
postMu = muFn(xs) + Ks'*Ki*(ftrain - muFn(Xtrain));
postCov = Kss - Ks'*Ki*Ks;

%% plot samples from posterior predictive
figure; hold on
% plot marginal posterior variance as gray band
mu = postMu(:);
S2 = diag(postCov);
f = [mu+2*sqrt(S2);flip(mu-2*sqrt(S2),1)];
fill([xs; flip(xs,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);

for i=1:3
  model = struct('mu', postMu(:), 'Sigma', postCov);
  fs = gauss_sample(model, 1);
  plot(xs, fs, 'k-', 'linewidth', 2)
  plot(Xtrain, ftrain, 'kx', 'markersize', 12, 'linewidth', 3);
end
title('samples from GP posterior');

end


function S = gauss_sample(model, n)
% Returns n samples from a multivariate Gaussian distribution
% S = AZ + mu

mu = model.mu;
Sigma = model.Sigma;

A = chol(Sigma, 'lower');
Z = randn(length(mu), n);
S = bsxfun(@plus, mu(:), A*Z)';
end
