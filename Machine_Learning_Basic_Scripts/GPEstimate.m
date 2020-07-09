function [mu] = GPEstimate(xPost,xPrior,hyperParameters,betaBar,h,y)
    xPost  = xPost(:); % change from row to vector or vector to vector
    xPrior = xPrior(:);
    y = y(:);
    N = length(xPrior);
    n = length(xPost);
    H = h(xPrior)';
    %% Ks, K and K
    k = GPSEKernel(xPrior,xPrior,hyperParameters(1));
    k_s = hyperParameters(3)*GPSEKernel(xPrior,xPost,hyperParameters(1));
    Ky = hyperParameters(3)*k+hyperParameters(2)*eye(N);
    %% posterior
    R = h(xPost)'-H*inv(Ky)*k_s;
    k_ss = hyperParameters(3)*GPSEKernel(xPost,xPost,hyperParameters(1)) + R'*inv(H*inv(Ky)*H')*R;
    
    %% L
    L = chol(Ky,'lower');
    Lk = L \ k_s;
%     Lss_post = chol(k_ss-Lk'*Lk,'lower');
    
    %% post. mu and y
    mu = (Lk') * (L \ y)+R'*betaBar;
%     Ys = mu+Lss_post*xPost;
end

