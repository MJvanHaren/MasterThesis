function val = GPSEKernel(a,b,l)
    % a = x data direction 1
    % b = x data direction 2
    % l = length parameter
    %% calculation
    a = reshape(a, [],1 );
    b = reshape(b, [],1 );
%     [D, n] = size(a);
%     [d, m] = size(b);
%     sqdist = repmat(a.^2,1,m) + repmat((b.^2)',n,1) - 2*a*b';
    sqdist = (repmat(a,size(b'))-repmat(b',size(a))).^2;
%     val = exp(-0.5/l^2*sqdist);
    val = exp(-0.5/l*sqdist);
%     K(p,q) = exp(-0.5*(repmat(p',size(q))-repmat(q,size(p'))).^2)
end
