function [Xnx, betaN,fn] = FFbeam(L,W,Th,E,Ro,HMMS,CS)
% function FFbeam(~)
% FFbeam.m Free-free end beam evaluations
% HELP:  This script computes mode shapes and corresponding natural
% frequencies of the free-free beam by a user specified mechanical
% properties and geometry size of the beam by using Euler-Bernoulli beam
% theory.
% Prepare the followings:
%  - Material properties of the beam, viz. density (Ro), Young's modulus (E)
%  - Specify a cross section of the beam, viz. square,rectangular, circular
%  - Geometry parameters of the beam, viz. Length, width, thickness
%  - 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 by Sulaymon L. ESHKABILOV, Ph.D
%                         October, 2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% display('What is the X-section of the beam?')
% disp('If circle, enter 1. If square, enter 2 ')
% disp('If rectangle, enter 3 ')
% disp('If your beam"s X-section is not listed here, enter 4')
% disp('To see an example #1 (rectangular X-section), hit ENTER or enter 0')
% disp('To see an example #2 (rectangular X-section thin beam), enter 5')
% CS=input(' Enter your choice:   ');

if isempty(CS) || CS==0
    disp('Example #1. Rectangular X-section Aluminum beam')
    disp('Length=0.321 [m], Width=0.05 [m], Thickness=0.006 [m];')
    disp('E=69.9*1e9 [Pa]; Ro=2770 [kg/m^3]')
    L=.321;
    W=.05;
    Th=.006;
    A=W*Th;
    V=L*W*Th;
    Ix=(1/12)*W*Th^3;
    E=69.9e+9;
    Ro=2770;
elseif CS==1
%     R=input('Enter Radius of the X-section in [m]:  ');
%     L=input('Enter Length in [m]:  ');
    Ix=(1/4)*pi*R^4;
    A=pi*R^2;
%     disp('Material proprties of the beam')
%     display('Do you know your beam"s material properties, viz. Young"s modulus and density ?')
%     YA=input('Enter 1, if you do; enter 0, if you don"t ');
%     if YA==1
%     E=input('Enter Young"s modulus in [Pa]:   ');
%     Ro=input('Enter material density in [kg/m^3]:   ');
%     else
%         display('Steel: E=2.1e+11 Pa; Ro=7850 [Kg/m^3]')
%         display('Copper: E=1.2e+11 Pa; Ro=8933 [Kg/m^3]')
%         display('Aluminum: E=0.69e+11 Pa; Ro=2700 [Kg/m^3]')
%         E=input('Enter Young"s modulus in [Pa]:  ');
%         Ro=input('Enter material density in [kg/m^3]:   ');
%     end
elseif CS==2
%     W=input('Enter Width of the X-section in [m]:  ');
%     L=input('Enter Length in [m]: ');
    Ix=(1/12)*W^4;
    A=W^2;
%     disp('Material proprties of the beam')
%     display('Do you know your beam"s material properties, viz. Young"s modulus and density ?')
%     YA=input('Enter 1, if you do; enter 0, if you don"t ');
%     if YA==1
%     E=input('Enter Young"s modulus in [Pa]:  ');
%     Ro=input('Enter material density in [kg/m^3]:  ');
%     else
%         display('Steel: E=2.1e+11 Pa; Ro=7850 [Kg/m^3]')
%         disprlay('Copper: E=1.2e+11 Pa; Ro=8933 [Kg/m^3]')
%         display('Aluminum: E=0.69e+11 Pa; Ro=2700 [Kg/m^3]')
%         E=input('Enter Young"s modulus in [Pa]:  ');
%         Ro=input('Enter material density in [kg/m^3]:  ');
%     end
elseif CS==3
%     W=input('Enter Width of the X-section in [m]:  ');
%     Th=input('Enter Thickness of the X-section in [m]:  ');
%     L=input('Enter Length in [m]:   ');
    Ix=(1/12)*W*Th^3;
    A=W*Th;
%     disp('Material properties of the beam')
%     display('Do you know your beam"s material properties, viz. Young"s modulus and density ?')
%     YA=input('Enter 1, if you do; enter 0, if you don"t ');
%     if YA==1
%     E=input('Enter Young"s modulus in [Pa]:  ');
% % %     Ro=input('Enter material density in [kg/m^3]:  ');
%     else
%         display('Steel: E=2.1e+11 Pa; Ro=7850 [Kg/m^3]  ')
%         display('Copper: E=1.2e+11 Pa; Ro=8933 [Kg/m^3]  ')
%         display('Aluminum: E=0.69e+11 Pa; Ro=2700 [Kg/m^3]  ')
%         E=input('Enter Young"s modulus in [Pa]:  ');
%         Ro=input('Enter material density in [kg/m^3]:  ');
%     end
elseif  CS==5 
    display('Example #2');
    display('It is a rectangular X-section Aluminum beam. ');
    display('Length=0.03 m; Width=0.005 m; Thickness=0.0005 m;')
    L=.03; W=.005; Th=.0005;
    A=W*Th;
    Ix=(1/12)*W*Th^3;
    E=70*1e9; Ro=2.7*1e3;
else
    F=warndlg('It is not clear what your choice of X-section of a beam is. Re-run the script and enter your beam"s X-section !!!','!! Warning !!');
    waitfor(F)
    display('Type in:>> FFbeam')
    pause(3)
    return
    
end
%        
% display('How many modes and mode shapes would you like to evaluate ?');
% HMMS=input('Enter the number of modes and mode shapes to be computed:  ');
if HMMS>=7
    disp('   ')
    warning('NOTE: Up to 6 mode shapes (plots) are displayed via the script. Yet, using evaluated data (Xnx) of the script, more mode shapes can be plotted');
    disp('   ')
end
    Nm=3*HMMS;
    jj=1;
    while jj<=Nm;
        betaNL(jj)=fzero(@(betaNL)cos(betaNL)*cosh(betaNL)-1,jj+3);
        jj=jj+3;
    end
    
    index=(betaNL~=0);
    betaNLall=(betaNL(index))';
    %fprintf('betaNL value is %2.3f\n', betaNLall);
    betaN=(betaNLall/L)';
k=1;
wn=zeros(1,length(betaN));
fn=ones(1,length(wn));
while k<=length(betaN);
    wn(k)=betaN(k)^2*sqrt((E*Ix)/(Ro*A));
    fn(k)=wn(k)/(2*pi);
    fprintf('Mode shape # %2f corresponds to nat. freq (fn): %3.3f\n', k, fn(k) )
    k=k+1;
end


x=linspace(0, L, 721);
sigmaN=zeros(1,HMMS);
for ii=1:HMMS;
    sigmaN(ii)=(cosh(betaNLall(ii))-cos(betaNLall(ii)))/(sinh(betaNLall(ii))-sin(betaNLall(ii)));
end
xl=x./L;

Tc='(cosh(betaN(ii).*x(jj))+cos(betaN(ii).*x(jj)))-sigmaN(ii).*(sinh(betaN(ii).*x(jj))+sin(betaN(ii)*x(jj)))';
Xnx=zeros(length(betaN),length(x));

for ii=1:length(betaN)
    for jj=1:length(x)
        Xnx(ii,jj)=eval(Tc);
    end
end
XnxMAX=max(abs(Xnx(1,1:end)));
Xnx=Xnx./XnxMAX;
% Plot mode shapes are arbitrarily normalized to unity;
disp('Note: first two mode shapes are not included are X0=const(translational)')
disp('and X0=A(x-l/2) (rotational)')
display('NOTE: Upto 5 mode shapes are displayed via the script options.') 
disp(' Yet, using evaluated data (Xnx) of the script, more mode shapes can be plotted')
MMS=HMMS;

% end

end

