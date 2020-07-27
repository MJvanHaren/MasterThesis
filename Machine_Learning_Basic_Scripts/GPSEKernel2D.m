function val = GPSEKernel2D(a,b,l)
    % a = x data direction 1
    % b = x data direction 2
    % c = y data direction 1
    % d = y data direction 2
    % l = length parameter (2D)
    %%
    x1 = a(:,1);
    x2 = a(:,2);
    x1p = b(:,1);
    x2p = b(:,2);

    %% calcs
    sqdist1 = (repmat(x1,size(x1p'))-repmat(x1p',size(x1))).^2;
    sqdist2 = (repmat(x2,size(x2p'))-repmat(x2p',size(x2))).^2;
    val = exp(-0.5/l(1)*sqdist1-0.5/l(2)*sqdist2);
end
