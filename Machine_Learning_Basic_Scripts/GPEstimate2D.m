function [mu] = GPEstimate2D(xPost,xPrior,hyperParameters,betaBar,h,y)
    y = y(:);
    N = length(xPrior);
    n = length(xPost);
    H = h(xPrior(:,1),xPrior(:,2))';
    %% Ks, K and K
    k = GPSEKernel2D(xPrior,xPrior,hyperParameters(1:2));
    k_s = hyperParameters(4)*GPSEKernel2D(xPrior,xPost,hyperParameters(1:2));
    Ky = hyperParameters(4)*k+hyperParameters(3)*eye(N);
    %% posterior
    R = h(xPost(:,1),xPost(:,2))'-H*inv(Ky)*k_s;    
    %% L
    L = chol(Ky,'lower');
    Lk = L \ k_s;
        %% post. mu and y
    mu = (Lk') * (L \ y)+R'*betaBar;
end

