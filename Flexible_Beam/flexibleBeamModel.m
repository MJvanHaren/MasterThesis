close all; 
clear all; 
clc;
%% style and paths
addpath(genpath('C:\Users\maxva\Google Drive\Documenten\TuE\Master\Thesis\MasterThesis'));    % for auxilary functions (GP/plot style etc.)
SetPlotLatexStyle();
[c1, c2, c3, c4, c5,c6,c7] = MatlabDefaultPlotColors();
%% definitions
L = 0.5;                                                                    % [m], length of free-free beam
W = 40e-3;                                                                  % [m], width (or height) of free-free beam
Th = 2e-3;                                                                  % [m], thickness of free-free beam
E = 2.1e11;                                                                 % [Pa], young's modulus of material (steel)
Ro = 7850;                                                                  % [Kg/m^3], density of material (steel)
HMMS = 5;                                                                   % [-], how many mode shapes, amount of modeshapes to estimate using euler equations
CS=3;                                                                       % [-], shape of beam, 3 means rectangular

Ix = [1 361 721];                                                           % indices on beam for Gy and Gz estimation. 1 means beginning of beam, 721 is the end
indices = [361 110 220 550 660 720];                                     % indices to perform ILC with BF to estimate FF parameters
newIndices = [165 455 605 690];                                          % indices to perform GP/nearest neighbour on 
N = length(indices);

noPriors = 100;                                                             % number of priors for GP regression
h = @(x) [ones(length(x(:)),1) x(:) x(:).^2 x(:).^3 x(:).^4 x(:).^5];       % basis for mean function, use prior model knowledge for this(!)
% h = @(x) [ones(length(x(:)),1) x(:) x(:).^2];
series = 1;                                                                 % 1=Do least squares before GP to determine mean function. 0=determine mean function using optimization in parallel with hyper parameter optimization

N_trials_ILC =3;                                                            % amount of trials done in ILC
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
    for r = 1:2
        G{i} = G{i}+(W(r,Ix(i))*P(r))/s^2;
    end
    for r = 3:R
        G{i} = G{i}+(W(r,Ix(i))*P(r))/(s^2+omegaList(r)^2+2*zeta(r)*s);
    end
end
Gz = G{2};
Gy = 0.5*(G{1}+G{end});
%% ILC
for i = 1:N
    [thetaGrid(:,i),Gu,history(i,:)] = FlexibleBeamILCBF(indices(i),toeplitzc,indx,N_trials_ILC,0,W,P,omegaList,zeta);
    close gcf;
end
%% GP
close all;
[mu, xprior,~,hyperParameters,betaBar] = GPRegressionFlexibleBeam(noPriors,m,N,X(indices),thetaGrid,h,series,indx);
%% resampling with  GP and others
newTheta = zeros(m,length(newIndices));
GPTheta = zeros(m,length(newIndices));
for j = 1:m
    GPTheta(j,:) = GPEstimate(X(newIndices),X(indices),hyperParameters(:,j),betaBar(:,j),h,thetaGrid(j,:));
end
for i = 1:length(newIndices)
    [newTheta(:,i),~,historyGP(i,:)] = FlexibleBeamILCBF(newIndices(i),toeplitzc,indx,N_trials_ILC,0,W,P,omegaList,zeta,GPTheta(:,i)');
    [~,index] = min(abs(indices-newIndices(i)));
    [~,~,historyBF(i,:)] = FlexibleBeamILCBF(newIndices(i),toeplitzc,indx,N_trials_ILC,0,W,P,omegaList,zeta,thetaGrid(:,index)');
    [~,~,historyBF2(i,:)] = FlexibleBeamILCBF(newIndices(i),toeplitzc,indx,N_trials_ILC,0,W,P,omegaList,zeta,thetaGrid(:,1)');
end
%%
figure
for i = 1:N
    p1 = semilogy(X(indices(i)),history(i,:).eNorm(1,end),'+','Color',c1,'Markersize',15); hold on;
    p2 = semilogy(X(indices(i)),history(i,:).eNorm(1,1),'p','Color',c6,'Markersize',10); 
end
for i = 1:length(newIndices)
    p3 = semilogy(X(newIndices(i)),historyGP(i,:).eNorm(1,1),'s','Color',c2,'MarkerFaceColor',c2,'Markersize',10);
    p4 = semilogy(X(newIndices(i)),historyBF(i,:).eNorm(1,1),'.','Color',c4,'Markersize',30);
    p5 = semilogy(X(newIndices(i)),historyBF2(i,:).eNorm(1,1),'^','Color',c5,'MarkerFaceColor',c5,'Markersize',10);
end


xlabel('Position x on free-free beam $[m]$');
ylabel('$\|e\|_2 [m^2]$');
legend([p1 p2 p3 p4 p5],{'Training data with converged FF parameters','No feedforward','FF parameters from GP','Using FF parameters from nearest neighbour converged ILC','Using FF parameters converged ILC at 0.5m'},'Location','northoutside')
%%
figure
for i = 1:N
    p1 = semilogy(X(indices(i)),history(i,:).eInfNorm(1,end),'+','Color',c1,'Markersize',15); hold on;
    p2 = semilogy(X(indices(i)),history(i,:).eInfNorm(1,1),'p','Color',c6,'Markersize',10);
end
for i = 1:length(newIndices)
    p3 = semilogy(X(newIndices(i)),historyGP(i,:).eInfNorm(1,1),'s','Color',c2,'MarkerFaceColor',c2,'Markersize',10);
    p4 = semilogy(X(newIndices(i)),historyBF(i,:).eInfNorm(1,1),'.','Color',c4,'Markersize',30);
    p5 = semilogy(X(newIndices(i)),historyBF2(i,:).eInfNorm(1,1),'^','Color',c5,'MarkerFaceColor',c5,'Markersize',10);
end

xlabel('Position x on free-free beam $[m]$');
ylabel('$\|e\|_\infty [m]$');
legend([p1 p2 p3 p4 p5],{'Training data with converged FF parameters','No feedforward','FF parameters from GP','Using FF parameters from nearest neighbour converged ILC','Using FF parameters determined using ILC at 0.25m'},'Location','northoutside');
%% re-doing ILC with BF at newIndices to see performance of GP
for i = 1:length(newIndices)
    [newThetaGrid(:,i),~,~] = FlexibleBeamILCBF(newIndices(i),toeplitzc,indx,N_trials_ILC,0,W,P,omegaList,zeta);
end
%% plotting
figure
for i =1:m
    subplot(floor(sqrt(m)),ceil(sqrt(m)),i);
    plot(X(indices),thetaGrid(i,:),'+','Color',c7,'Markersize',15); hold on;
    plot(X(newIndices),newThetaGrid(i,:),'^','Color',c1,'MarkerFaceColor',c1,'Markersize',10);
    plot(X(newIndices),GPTheta(i,:),'s','Color',c2,'MarkerFaceColor',c2,'Markersize',10);

    xlabel('Position on free-free beam $[m]$');
    ylabel('Feedforward parameter [var]');
end
legend('Training data (FF parameters from ILC)','True feedforward parameters determined using ILC','Estimated FF parameters with GP');