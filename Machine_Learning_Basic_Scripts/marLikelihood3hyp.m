function [minlogp,dlogpdtheta] = marLikelihood3hyp(xT,y,unk)
    % xT    = x data of training data y (N points)
    % y     = training data (N points)
    % h     = mean function
    % unk   = vector of unknown (hyper)parameters (3*1) (L,sigman,sigmaf)
    %% definitions
    N = length(y);
    xT = reshape(xT, [],1);
    %% kernel K
    k = GPSEKernel(xT,xT,unk(1));       % kernel with only length parameter
    Ky = unk(3)^2*k + unk(2)^2*eye(N);  % kernel with added noise hyperparameter

    %% GP 4 ML (5.9)
    % invKy = inv(Ky);                  % GP 4 ML (5.9)
    % alpha =  invKy*y;                 % GP 4 ML (5.9)
    % minlogp = ( 0.5*y'*invKy*y...
    %             + 0.5*log(det(Ky))...
    %             + n/2*log(2*pi));       % GP 4 ML (2.30)

    %% GP 4 ML algorithm p.19
    L = chol(Ky,'lower');               % cholesky decomposition
    alphaA = L'\(L\y);                  % GP 4 ML algoritme (p.19)
    minlogp = ( 0.5*y'*alphaA...
        +(sum(log(diag(L))))...
        +N/2*log(2*pi));                % GP 4 ML algoritme (p.19)

    %% derivatives of log likelhood w.r.t. hyper parameters 
    sqDist = repmat(xT.^2,1,N) + repmat((xT.^2)',N,1) - 2*(xT)*xT';
    dKtheta1 = unk(3)^2*k*sqDist/unk(1)^3;                                   % https://bit.ly/2WR9e6i
    dKtheta2 = 2*unk(2)*eye(N);                                     % slightly different w.r.t. Gaussian process regresion techniques (3.13)
    dKtheta3 = 2*unk(3)*k;
    dlogpdtheta(1) = -0.5*trace((alphaA*alphaA'-Ky)*dKtheta1);      % GP 4 ML (5.9)
    dlogpdtheta(2) = -0.5*trace((alphaA*alphaA'-Ky)*dKtheta2);      % GP 4 ML (5.9)
    dlogpdtheta(3) = -0.5*trace((alphaA*alphaA'-Ky)*dKtheta3);      % GP 4 ML (5.9)
end

