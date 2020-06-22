close all; 
% clear all; 
clc
%% definitions
s = tf('s');
omegaList = [2 6 fn]*2*pi;
zeta = [5 3 betaN];
R = length(omegaList);
P = ones(R,1);
% x = 0.1; % [0 0.5]
X = linspace(0,0.4,720);
Lx = length(X);
Ix = 100;
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
bodemag(G);