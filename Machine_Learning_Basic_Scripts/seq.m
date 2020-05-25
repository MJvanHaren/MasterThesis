z = 4.97;

randn('seed',3)
rand('seed',4)

h = figure(1);
clf
set(gca,'FontSize',24)

K = inline('exp(-0.5*(repmat(p'',size(q))-repmat(q,size(p''))).^2)');
xs = [-5:0.1:5]'; m = length(xs); tiny = 1e-6;

x = []; 
f = [];
n = 0;
for i=1:20
  hold off
  mu = K(x,xs)*inv(K(x,x)+tiny*eye(n))*f;
  if i>1
    S = K(xs,xs)+tiny*eye(m)-K(x,xs)*inv(K(x,x)+tiny*eye(n))*K(xs,x);  
  else
    S = K(xs,xs)+tiny*eye(m);
  end
  xxs = max(min([xs; flipdim(xs,1)],z),-z);
  if i>1
    ys = max(min([mu+2*sqrt(diag(S));flipdim(mu-2*sqrt(diag(S)),1)],z/2),-z/2);
  else
    ys = max(min([2*sqrt(diag(S));flipdim(-2*sqrt(diag(S)),1)],z/2),-z/2);
  end
  fill(xxs,ys,[7 7 7]/8, 'EdgeColor', 0.3*ones(3,1))
  hold on

  X = 10*rand-5;  
  mu = K(x,X)*inv(K(x,x)+tiny*eye(n))*f;
  s2 = K(X,X)+tiny-K(x,X)*inv(K(x,x)+tiny*eye(n))*K(X,x);
  if i>1
    F = mu + sqrt(s2)*randn;
  else
    F = randn;
  end
  plot(x,f,'bx','MarkerSize',32,'LineWidth',2)
   plot([X X],[-2.5 2.5],'r-')
  pause 

  plot(X,F,'rx','MarkerSize',32,'LineWidth',2)


  x(i,1) = X;
  f(i,1) = F;
  n = i;

  % allow switching off printing the figures
  if 0
    % pretty format the figure
    set(h,'PaperOrientation','landscape','PaperPosition',[0 0 29.7 21.0],'PaperUnits','centimeters','PaperType','A4');
    % output to pdf
    print (h, '-dpdf', sprintf('~/svn/teaching/4f13/1112/figures/seq_fullGP_M%d.pdf',i));
  end

  pause

end
