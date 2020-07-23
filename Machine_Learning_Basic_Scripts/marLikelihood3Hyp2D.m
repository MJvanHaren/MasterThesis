function [minlogp,dlogpdtheta] = marLikelihood3Hyp2D(xT,y,unk)
    % xT    = x data of training data y (N points)
    % y     = training data (N points)
    % unk   = vector of unknown (hyper)parameters (3*1) (L,sigman,sigmaf)
    %% definitions
    N = length(y);
%     xT = xT(:);
    %% kernel K
    k = GPSEKernel2D(xT,xT,unk(1:2));       % kernel with only length parameter
    Ky = unk(4)*k + unk(3)*eye(N);  % kernel with added noise hyperparameter

    %% GP 4 ML algorithm p.19
    L = chol(Ky,'lower');               % cholesky decomposition
    alphaA = L'\(L\y);                  % GP 4 ML algoritme (p.19)
    minlogp = ( 0.5*y'*alphaA...
        +(sum(log(diag(L))))...
        +N/2*log(2*pi));                % GP 4 ML algoritme (p.19)
end

