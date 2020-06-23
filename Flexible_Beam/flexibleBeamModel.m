close all; 
clear all; 
clc
%% definitions
L = 0.5; %m
W = 40e-3; %m
Th = 2e-3; %m
E = 2.1e11;
Ro = 7850;
HMMS = 3;
CS=3; %rectangular
Ix = 350;
%% execution
[Xnx, betaN, fn] = FFbeam(L,W,Th,E,Ro,HMMS,CS);
s = tf('s');
omegaList = [2 6 fn]*2*pi;
zeta = [3 5e-1 betaN];
R = length(omegaList);
P = 5e-2*ones(R,1);
X = linspace(0,L,size(Xnx,2));
Lx = length(X);
% Ix = ceil(Lx/2);
x = X(Ix);
W = zeros(R,Lx);
W(1,:) = ones(1,Lx);
W(2,:) = linspace(-1,1,Lx);
W(3:end,:) = Xnx;
%% iterating over modes
G = 0;
for r = 1:R
    G = G+(W(r,Ix)*P(r))/(s^2+omegaList(r)^2+2*zeta(r)*s);
end
options = bodeoptions;
options.FreqUnits = 'Hz'; 
bode(G,options);
xlim([8e-1 5e2]);