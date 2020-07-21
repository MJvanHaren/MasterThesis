function val = GPSEKernel(a,b,l)
    % a = x data direction 1
    % b = x data direction 2
    % l = length parameter
    %% calculation
    a = a(:);
    b = b(:);

    sqdist = (repmat(a,size(b'))-repmat(b',size(a))).^2;
    val = exp(-0.5/l*sqdist);
end
