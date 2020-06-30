close all; 
clear all; 
clc;
addpath('C:\Users\maxva\Google Drive\Documenten\TuE\Master\Thesis\MasterThesis\Machine_Learning_Basic_Scripts');    % for RBF kernel
%% definitions
L = 0.5; %m
W = 40e-3; %m
Th = 2e-3; %m
E = 2.1e11;
Ro = 7850;
HMMS = 5;
CS=3; %rectangular
Ix = [1 361 721]; %[1 721]
indices = [361 10 110 220 550 660 705];
newIndices = [60 180 375 580 690];
N = length(indices);
%GP
noPriors = 200;
h = @(x) [ones(length(x(:)),1) x(:) x(:).^2 x(:).^3 x(:).^4 x(:).^5];    % basis for mean function
% h = @(x) [ones(size(x(:),1),1)];
series = 1; 
N_trials_ILC =6;
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

%% execution
[Xnx, betaN, fn] = FFbeam(L,W,Th,E,Ro,HMMS,CS);
s = tf('s');
omegaList = [4 10 fn]*2*pi;
zeta = [5 5 betaN];
R = length(omegaList);
P = 5e3*ones(R,1);
X = linspace(0,L,size(Xnx,2));
Lx = length(X);
% Ix = ceil(Lx/2);
W = zeros(R,Lx);
W(1,:) = ones(1,Lx);
W(2,:) = linspace(-1,1,Lx);
W(3:end,:) = Xnx;



%% iterating over modes and positions
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
%     figure(2)
%     bodemag(G{i},options); hold on;
end
Gz = G{2};
figure(2);
bode(Gz);hold on;
Gy = 0.5*(G{1}+G{end});
bode(Gy);
legend('$G\_{z}$','$G\_{y}$','Interpreter','Latex','FontSize',14)
Gsys = Gy;
%% ILC
for i = 1:N
    [theta_grid(:,i),Gu,history(i,:)] = FlexibleBeamILCBF(indices(i),toeplitzc,indx,N_trials_ILC,0,W,P,omegaList,zeta);
    close gcf;
end
%%
figure
plot(X(indices),theta_grid(1,:),'s');
figure
plot(X(indices),theta_grid(2,:),'s');
%% GP
close all
[mu, xprior,~,hyperParameters,betaBar] = GPregression(noPriors,m,N,X(indices),theta_grid,h,series,indx);
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


xlabel('Position x on free-free beam [m]');
ylabel('||e||_2 [m^2]');
legend([p1 p2 p3 p4],{'Training data with converged FF parameters','FF parameters from GP','Using FF parameters from nearest neighbour converged ILC','Using FF parameters converged ILC at 0.5m'},'Location','best')
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

xlabel('Position x on free-free beam [m]');
ylabel('||e||_? [m^2]');
legend([p1 p2 p3 p4],{'Training data with converged FF parameters','FF parameters from GP','Using FF parameters from nearest neighbour converged ILC','Using FF parameters converged ILC at 0.5m'},'Location','best')