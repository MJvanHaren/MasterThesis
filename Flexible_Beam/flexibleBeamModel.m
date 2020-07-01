close all; 
clear all; 
clc;
%%
addpath(genpath('C:\Users\maxva\Google Drive\Documenten\TuE\Master\Thesis\MasterThesis'));    % for auxilary functions (GP/plot style etc.)
SetPlotLatexStyle();
%% definitions
L = 0.5;                                                                    % [m], length of free-free beam
W = 40e-3;                                                                  % [m], width (or height) of free-free beam
Th = 2e-3;                                                                  % [m], thickness of free-free beam
E = 2.1e11;                                                                 % [Pa], young's modulus of material (steel)
Ro = 7850;                                                                  % [Kg/m^3], density of material (steel)
HMMS = 5;                                                                   % [-], how many mode shapes, amount of modeshapes to estimate using euler equations
CS=3;                                                                       % [-], shape of beam, 3 means rectangular

Ix = [1 361 721];                                                           % indices on beam for Gy and Gz estimation. 1 means beginning of beam, 721 is the end
indices = [361 10 110 220 550 660 705];                                     % indices to perform ILC with BF to estimate FF parameters
newIndices = [60 180 375 580 690];                                          % indices to perform GP/nearest neighbour on 
N = length(indices);

noPriors = 200;                                                             % number of priors for GP regression
h = @(x) [ones(length(x(:)),1) x(:) x(:).^2 x(:).^3 x(:).^4 x(:).^5];       % basis for mean function, use prior model knowledge for this(!)
series = 1;                                                                 % 1=Do least squares before GP to determine mean function. 0=determine mean function using optimization in parallel with hyper parameter optimization

N_trials_ILC =6;                                                            % amount of trials done in ILC
%% UI
methodLQ = questdlg('constructing L & Q from:', ...
    'constructing L & Q from:', ...
    'toeplitz impulse response matrix', 'Efficient way of calculating L & Q', 'Efficient way of calculating L & Q');

switch methodLQ
    case 'toeplitz impulse response matrix'
        toeplitzc = 1;
    case 'Efficient way of calculating L & Q'
        toeplitzc = 0;
end

list = {'Position','Sign Velocity','Velocity','Acceleration','Jerk','Snap'};
[indx,~] = listdlg('ListString',list);
m = length(indx); % amount of basises used

%% Determine (normalized) modeshapes, damping of modes, freqency of modes and *set* gain matrix P
[Xnx, betaN, fn] = FFbeam(L,W,Th,E,Ro,HMMS,CS);
s = tf('s');
omegaList = [4 10 fn]*2*pi;
zeta = [5 5 betaN];
R = length(omegaList);
P = 5e3*ones(R,1);
X = linspace(0,L,size(Xnx,2));
Lx = length(X);
W = zeros(R,Lx);
W(1,:) = ones(1,Lx);
W(2,:) = linspace(-1,1,Lx);
W(3:end,:) = Xnx;
%% iterating over modes and positions to determine Gy and Gz
options = bodeoptions;
options.FreqUnits = 'Hz'; 
options.MagUnits = 'dB';
options.Ylim = [-100 50];
options.Xlim = [8e-1 8e2];
for i = 1:length(Ix)
    x(i) = X(Ix(i));
    G{i} = 0;
    for r = 1:R
        G{i} = G{i}+(W(r,Ix(i))*P(r))/(s^2+omegaList(r)^2+2*zeta(r)*s);
    end
end
Gz = G{2};
Gy = 0.5*(G{1}+G{end});
%% ILC
for i = 1:N
    [theta_grid(:,i),Gu,history(i,:)] = FlexibleBeamILCBF(indices(i),toeplitzc,indx,N_trials_ILC,0,W,P,omegaList,zeta);
    close gcf;
end
%% GP
close all;
[mu, xprior,~,hyperParameters,betaBar] = GPRegressionFlexibleBeam(noPriors,m,N,X(indices),theta_grid,h,series,indx);
%% resampling with  GP and others
newTheta = zeros(m,length(newIndices));
GPTheta = zeros(m,length(newIndices));
for j = 1:m
    GPTheta(j,:) = GPEstimate(X(newIndices),X(indices),hyperParameters(:,j),betaBar(:,j),h,theta_grid(j,:));
end
for i = 1:length(newIndices)
    [newTheta(:,i),~,historyGP(i,:)] = FlexibleBeamILCBF(newIndices(i),toeplitzc,indx,N_trials_ILC,0,W,P,omegaList,zeta,GPTheta(:,i)');
    [~,index] = min(abs(indices-newIndices(i)));
    [~,~,historyBF(i,:)] = FlexibleBeamILCBF(newIndices(i),toeplitzc,indx,N_trials_ILC,0,W,P,omegaList,zeta,theta_grid(:,index)');
    [~,~,historyBF2(i,:)] = FlexibleBeamILCBF(newIndices(i),toeplitzc,indx,N_trials_ILC,0,W,P,omegaList,zeta,theta_grid(:,1)');
end
%%
figure
for i = 1:N
    p1 = plot(X(indices(i)),history(i,:).eNorm(1,end),'+','Color',	[0, 0.4470, 0.7410],'Markersize',15); hold on;
end
for i = 1:length(newIndices)
    p2 = plot(X(newIndices(i)),historyGP(i,:).eNorm(1,1),'s','Color',	[0.8500, 0.3250, 0.0980],'MarkerFaceColor',[0.8500, 0.3250, 0.0980],'Markersize',10);
    p3 = plot(X(newIndices(i)),historyBF(i,:).eNorm(1,1),'.','Color',	[0.4940, 0.1840, 0.5560],'Markersize',30);
    p4 = plot(X(newIndices(i)),historyBF2(i,:).eNorm(1,1),'^','Color',	[0.4660, 0.6740, 0.1880],'MarkerFaceColor',[0.4660, 0.6740, 0.1880],'Markersize',10);
end


xlabel('Position x on free-free beam $[m]$');
ylabel('$\|e\|_2 [m^2]$');
legend([p1 p2 p3 p4],{'Training data with converged FF parameters','FF parameters from GP','Using FF parameters from nearest neighbour converged ILC','Using FF parameters converged ILC at 0.5m'},'Location','northoutside')
%%
figure
for i = 1:N
    p1 = plot(X(indices(i)),history(i,:).eInfNorm(1,end),'+','Color',	[0, 0.4470, 0.7410],'Markersize',15); hold on;
end
for i = 1:length(newIndices)
    p2 = plot(X(newIndices(i)),historyGP(i,:).eInfNorm(1,1),'s','Color',	[0.8500, 0.3250, 0.0980],'MarkerFaceColor',[0.8500, 0.3250, 0.0980],'Markersize',10);
    p3 = plot(X(newIndices(i)),historyBF(i,:).eInfNorm(1,1),'.','Color',	[0.4940, 0.1840, 0.5560],'Markersize',30);
    p4 = plot(X(newIndices(i)),historyBF2(i,:).eInfNorm(1,1),'^','Color',	[0.4660, 0.6740, 0.1880],'MarkerFaceColor',[0.4660, 0.6740, 0.1880],'Markersize',10);
end

xlabel('Position x on free-free beam $[m]$');
ylabel('$\|e\|_\infty [m]$');
legend([p1 p2 p3 p4],{'Training data with converged FF parameters','FF parameters from GP','Using FF parameters from nearest neighbour converged ILC','Using FF parameters determined using ILC at 0.25m'},'Location','northoutside');
%% re-doing ILC with BF at newIndices to see performance of GP
for i = 1:length(newIndices)
    [newThetaGrid(:,i),~,~] = FlexibleBeamILCBF(newIndices(i),toeplitzc,indx,N_trials_ILC,0,W,P,omegaList,zeta);
end
%% plotting
figure
for i =1:m
    subplot(floor(sqrt(m)),ceil(sqrt(m)),i)
    plot(X(newIndices),newThetaGrid(i,:),'+','Color',	[0, 0.4470, 0.7410],'Markersize',15); hold on;
    plot(X(newIndices),GPTheta(i,:),'s','Color',	[0.8500, 0.3250, 0.0980],'MarkerFaceColor',[0.8500, 0.3250, 0.0980],'Markersize',10);
    for ii = 1:length(newIndices)
        [~,index] = min(abs(indices-newIndices(ii)));
        nearestNeighbour(:,ii) = theta_grid(:,index);
    end
    plot(X(newIndices),nearestNeighbour(i,:),'.','Color',	[0.4940, 0.1840, 0.5560],'Markersize',30);
    
    xlabel('Position on free-free beam $[m]$');
    ylabel('Feedforward parameter [var]');
end
legend('True feedforward parameters determined using ILC','Estimated FF parameters with GP','Estimated FF parameters using nearest neighbour');