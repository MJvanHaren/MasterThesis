function [minlogp,dlogpdtheta] = marLikelihood3Hyp2DMeanFunc(xT,y,unk,betaBar,h,var)
    % xT    = x data of training data y (N points)
    % y     = training data (N points)
    % unk   = vector of unknown (hyper)parameters (3*1) (L,sigman,sigmaf)
    % betaBar = basis functions linear parameters
    % h = bais for mean functions
    %% definitions
    N = length(y);
    H = h(xT(:,1),xT(:,2));
    %% kernel K
    k = GPSEKernel2D(xT,xT,unk(1:2));       % kernel with only length parameter
    Ky = unk(3)*k + diag(var);  % kernel with added noise hyperparameter

    %% GP 4 ML algorithm p.19
    L = chol(Ky,'lower');               % cholesky decomposition
    alphaA = L'\(L\(y-H*betaBar));                  % GP 4 ML algoritme (p.19)
    minlogp = ( 0.5*(y-H*betaBar)'*alphaA...
        +(sum(log(diag(L))))...
        +N/2*log(2*pi));                % GP 4 ML algoritme (p.19)
end

