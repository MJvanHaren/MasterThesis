function [Ys,mu,stdv] = GPEstimateNoMean(xPost,xPrior,hyperParameters,y)
    xPost  = xPost(:); % change from row to vector or vector to vector
    xPrior = xPrior(:);
    y = y(:);
    N = length(xPrior);
    n = length(xPost);
    %% Ks, K and K
    k = GPSEKernel(xPrior,xPrior,hyperParameters(1));
    k_s = hyperParameters(3)*GPSEKernel(xPrior,xPost,hyperParameters(1));
    Ky = hyperParameters(3)*k+hyperParameters(2)*eye(N);
    %% posterior
    k_ss = hyperParameters(3)*GPSEKernel(xPost,xPost,hyperParameters(1));
    
    %% L
    L = chol(Ky,'lower');
    Lk = L \ k_s;
    Lss_post = chol(k_ss-Lk'*Lk+1e-8*eye(n),'lower');
    
    %% post. mu and y
    mu = (Lk') * (L \ y);
    Ys = mu+Lss_post*xPost;
    
    k_ss = hyperParameters(3)*GPSEKernel(xPost,xPost,hyperParameters(1));
    s2 = diag(k_ss)' - sum(Lk.^2,1);
    stdv = sqrt(s2);
end

