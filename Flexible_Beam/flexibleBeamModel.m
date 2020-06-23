close all; 
clear all; 
clc;
%% definitions
L = 0.5; %m
W = 40e-3; %m
Th = 2e-3; %m
E = 2.1e11;
Ro = 7850;
HMMS = 5;
CS=3; %rectangular
Ix = [1 361 721]; %[1 721]
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
zeta = [5 5e-1 betaN];
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
bodemag(Gz,options);hold on;
Gy = 0.5*(G{1}+G{end});
bodemag(Gy,options);
legend('$G\_{z}$','$G\_{y}$','Interpreter','Latex','FontSize',14)
Gsys = Gy;
load GyGzcontroller.mat
Ts = shapeit_data.C_tf_z.Ts;
%% ILC
[theta_jplus1,G,history] = FlexibleBeamILCBF(220,toeplitzc,indx,4,1,W,P,omegaList,zeta);

