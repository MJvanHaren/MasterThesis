function [minlogp,dlogpdtheta] = marLikelihood3HypMeanFunc(xT,y,unk,betaBar,h)
    % xT    = x data of training data y (N points)
    % y     = training data (N points)
    % h     = mean function
    % unk   = vector of unknown (hyper)parameters (3*1) (L,sigman,sigmaf)
    % betaBar = mean function linear values
    %% definitions
    N = length(y);
    xT = xT(:);
    %% kernel K
    k = GPSEKernel(xT,xT,unk(1));       % kernel with only length parameter
    Ky = unk(3)*k + unk(2)*eye(N);  % kernel with added noise hyperparameter

    %% GP 4 ML algorithm p.19
    L = chol(Ky,'lower');               % cholesky decomposition
    alphaA = L'\(L\(y-h(xT)*betaBar));                  % GP 4 ML algoritme (p.19)
    minlogp = ( 0.5*(y-h(xT)*betaBar)'*alphaA...
        +(sum(log(diag(L))))...
        +N/2*log(2*pi));                % GP 4 ML algoritme (p.19)
end

