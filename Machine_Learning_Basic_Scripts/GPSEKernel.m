function val = GPSEKernel(a,b,l)
    % a = x data direction 1
    % b = x data direction 2
    % l = length parameter
    %% calculation
    [D, n] = size(a);
    [d, m] = size(b);
    sqdist = repmat(a.^2,1,m) + repmat((b.^2)',n,1) - 2*a*b';
    val = exp(-0.5/l^2*sqdist);
end
