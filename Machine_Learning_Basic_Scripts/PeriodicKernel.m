function val = PeriodicKernel(a,b,p,l)
    % a = x data direction 1
    % b = x data direction 2
    %% calculation
    a = reshape(a, [],1 );
    b = reshape(b, [],1 );
    dist = (repmat(a,size(b'))-repmat(b',size(a)));
    val = exp(-2/l^2*(sin(pi*abs(dist)/p)).^2);
end

