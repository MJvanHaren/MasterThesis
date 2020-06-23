function [mu, xPrior, var] = GPregression(n,m,N,xTraining,yTraining,h,series)
    % n                 = amount of priors
    % m                 = amount of basises(functions) to regress
    % N                 = amound of training data
    % xTraining         = range of x values of training data
    % yTraining         = training data
    % h                 = basis for mean function
    %% definitions
    xstart = 0;
    xend = 0.5;
    xPrior = linspace(xstart,xend,n);
    mh = size(h(1),2);
     %% optimization of the hyper parameters ini
    x0 = [2;xTraining(1)*1e-1;7.5];             % initial guess for hyper parameters
    ub = [10000,1e-3,10000,10000];                   % lower and upper bounds for hyper parameters
    lb = [0.025,1e-15,0.025,10];

    nGrids = 12;  
    x01 = linspace(0.01,100,nGrids);
    x02 = logspace(-12,-5,nGrids);
    x03 = logspace(-2,2,nGrids);
    if series == 1 
        x04 = 1;
        hyp4 = 0;
    else
        x04 = linspace(50,2500,nGrids);
        hyp4 = 1;
    end
    [X01,X02,X03,X04] = ndgrid(x01,x02,x03,x04);

    fval = zeros(size(X01,1),size(X02,2),size(X03,3),size(X04,4));

    for i = 1:m
        y = yTraining(i,:)';
        for ii = 1:(size(X01,1))
            for j = 1:(size(X02,2))
                for ij = 1:size(X03,3)
                    for jj = 1:size(X04,4)
                      fval(ii,j,ij,jj) = marLikelihood4hyp(xTraining,y,h,[X01(ii,j,ij,jj); X02(ii,j,ij,jj); X03(ii,j,ij,jj); X04(ii,j,ij,jj)],hyp4);
                    end
                end
            end
        end
        if (length(size(X01)))<3
            figure(1);
            mini = (min(min(fval)));
            [I]=find(fval==mini);
            fval(fval >= mini+2*abs(mini)) = mini+2*abs(mini);
            subplot(floor(sqrt(m)),ceil(sqrt(m)),i);
            surf(X01,X02,fval,'LineStyle','none');
            xlabel('$l$','interpreter','Latex');
            ylabel('$\sigma_n$','interpreter','Latex')
            set(gca,'yscale','log');
        elseif (length(size(X01)))==3
            mini = min(min(min(fval)));
            [I]=find(fval==mini);clc
        else
            mini = min(min(min(min(fval))));
            [I]=find(fval==mini);
        end
        xres0 = [X01(I); X02(I);X03(I);X04(I)];
        options = optimoptions('fmincon','Display','off',...
                    'Algorithm','interior-point',...          % interior point does not work correctly with specifyobjectivegradient on
                    'SpecifyObjectiveGradient',false,...
                    'CheckGradients',false,...
                    'StepTolerance',1e-10);
        [xres,~] = fmincon(@(x) marLikelihood4hyp(xTraining,y,h,x,hyp4),xres0,[],[],[],[],lb,ub,[],options);

        meanfunc = [];
        covfunc = @covSEiso;                        % Squared Exponental covariance function
        likfunc = @likGauss;                        % Gaussian likelihood
        hyp = struct('mean', [], 'cov', log([x0(1) x0(3)]), 'lik', log(x0(2)));
        hypUpdate = minimize(hyp,@gp, -1000,@infGaussLik, meanfunc, covfunc, likfunc, xTraining', y);
        [mu2(:,i), var2(:,i)] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, xTraining', y, xPrior');
        xres2 = exp([hypUpdate.cov(1) hypUpdate.lik hypUpdate.cov(2)]);
        %% evaluation of kernel functions using optimized hyperparameters
        k = GPSEKernel(xTraining',xTraining',xres(1));
        k_s = xres(3)*GPSEKernel(xTraining',xPrior',xres(1));

        %L and Lk
        Ky = xres(3)*k+xres(2)*eye(N);
        L = chol(Ky,'lower');
        Lk = L \ k_s;
        
        % mean function
        B = xres(4)*eye(mh);
        H = h(xTraining)';
        if series == 1
            betaBar = inv(H*H')*H*yTraining(i,:)';
        else
            betaBar = inv(H*inv(Ky)*H')*H*inv(Ky)*y;
        end
        Hs = h(xPrior)';
        R = Hs-H*inv(Ky)*k_s;
        
        % kernel of prediction
        k_ss = xres(3)*GPSEKernel(xPrior',xPrior',xres(1)) + R'*inv(H*inv(Ky)*H')*R;
        
        % mu and SD/var
        mu(:,i) = (Lk') * (L \ y)+R'*betaBar;
        var(:,i) = (diag(k_ss)' - sum(Lk.^2,1))';

        %% plotting
% %         A = min([mu(:,i)-3*sqrt(var(:,i));mu2(:,i)-3*sqrt(var2(:,i))]);
%         B = max([mu(:,i)+3*sqrt(var(:,i));mu2(:,i)+3*sqrt(var2(:,i))]);
        figure(2);
        subplot(floor(sqrt(m)),2*ceil(sqrt(m)),2*(i-1)+2);
        f = [mu2(:,i)+3*sqrt(var2(:,i)); flipdim(mu2(:,i)-3*sqrt(var2(:,i)),1)];
        fill([xPrior'; flipdim(xPrior',1)], f, [7 7 7]/8)
        hold on; plot(xPrior', mu2(:,i)); plot(xTraining', y, '+','MarkerSize',10);
        xresMin = exp([hypUpdate.cov(1);hypUpdate.lik;hypUpdate.cov(2)]);
        title(['Basis #',num2str(i),' using minimize function']);
%         ylim([A B]);
        xlabel('Position xs [m]');
        ylabel('Feedforward parameter [var]');
        subplot(floor(sqrt(m)),2*ceil(sqrt(m)),2*(i-1)+1);
        inBetween = [(mu(:,i)+3*sqrt(var(:,i)))' fliplr((mu(:,i)-3*sqrt(var(:,i)))')];
        x2 = [xPrior, fliplr(xPrior)];
        fill(x2,inBetween, [7 7 7]/8);hold on;
        plot(xPrior,mu(:,i)); 
        plot(xTraining,y,'+','MarkerSize',10);
        
        title(['Basis #',num2str(i),' using grid+fmincon'])
        xlabel('Position xs [m]');
        ylabel('Snap Feedforward parameter [kgs^2]');
%         ylim([A B]);
    end
    legend('$\mu \pm 3\sigma$','$\mu$ of fitted posterior function','Generated samples','Interpreter','Latex')
end

