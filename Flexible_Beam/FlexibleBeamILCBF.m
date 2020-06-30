function [theta_jplus1,G,history] = FlexibleBeamILCBF(varargin)
    % Ix        = varargin{1}, position index x of H-bridge
    % toeplitz  = varargin{2}, choice of L and Q matrix calculation (1 = toeplitz impulse response, something else is efficient calc.)
    % basisList = varargin{3}, list of basisses used (1 = p, 2 = sign(v), 3 = v, 4 = a, 5 = j, 6 = s)
    % N_trial   = varargin{4}, number of trials in ILC with BF
    % plotToggle= varargin{5}, plot on (1) or off (0)
    % W         = varargin{6}, normalized mode shapes of beam
    % P         = varargin{7}, gain of mode shapes
    % omegaList = varargin{8}, list of frequencies of modes [rad/s]
    % zeta      = varargin{9}, list of damping for eigenmodes  
    % theta     = varargin{10} (OPTIONAL) initial value of theta for ILC with BF
    %% definitions
    Ix          = varargin{1};
    toeplitzc   = varargin{2};
    basisList   = varargin{3};
    N_trial     = varargin{4};
    plotToggle  = varargin{5};
    W           = varargin{6};
    P           = varargin{7};
    omegaList   = varargin{8};
    zeta        = varargin{9};
    %% model
    s = tf('s');
    G = 0;
    nR = length(omegaList);
    for r = 1:nR
        G = G+(W(r,Ix)*P(r))/(s^2+omegaList(r)^2+2*zeta(r)*s);
    end
%     figure(1)
%     bode(G);
    
    load('GyGzcontrollerWorse.mat');     % load controller synth. from shapeit
    C = shapeit_data.C_tf;
    CDT = shapeit_data.C_tf_z;
    Ts = shapeit_data.C_tf_z.Ts;
    
    Gss = c2d(ss(G),Ts);
    PS = feedback(Gss,CDT);                 % for simulating signals
    %% trajectory
%     [ttraj, ddx]  = make4(1,2,3,15,150,Ts);
    [ttraj, ddx]  = make4(0.5e-3,1,20,100,1500,Ts);
    [~,tx,d,j,a,v,p,~]=profile4(ttraj,ddx(1),Ts);

    
    yref = [p;zeros(40,1)+p(end);-p+p(end)];
    tin = 0:Ts:(length(yref)-1)*Ts;
    yrefTimeSeries = timeseries(yref,tin);
    N = length(yref);
    basisArray = [p sign(v) v a j d];
    psi = basisArray(:,basisList);
    mpsi = size(psi,2);
    
    %% figure
    if 0
        figure
        subplot(1,5,1)
        plot(tx,p)
        xlabel('Time [s]')
        ylabel('Position [m]')
        subplot(1,5,2)
        plot(tx,v)
        xlabel('Time [s]')
        ylabel('Velocity [m/s]')
        subplot(1,5,3)
        plot(tx,a)
        xlabel('Time [s]')
        ylabel('Acceleration [m/s^2]')
        subplot(1,5,4)
        plot(tx,j)
        xlabel('Time [s]')
        ylabel('Jerk [m/s^3]')
        subplot(1,5,5)
        plot(tx,d)
        xlabel('Time [s]')
        ylabel('Snap [m/s^4]')
   end
    %%  ILC BF
    we = eye(N)*1e6; %% I_N
    wf = eye(N)*0e-6;
    wDf = eye(N)*0e-6;

    if toeplitzc
        % Impulse response matrix J
        [PS_sysic_ss_den,PS_sysic_ss_num] = tfdata(PS,'v'); %tfdata(G_ss*S_d,'v');
        [h,t]   = dimpulse( PS_sysic_ss_den, PS_sysic_ss_num, N );
        J_ini  	= toeplitz( h, [h(1) , zeros(1,N-1)] );

        % Set small values to zero & SPARSE:
        J2  = J_ini;
        J2( abs(J2) < 1e-9 ) = 0;
        J   = (J2);

        % calculate filters
        R2 = inv(psi'*(J'*we*J+wf+wDf)*psi);
        L = R2*psi'*J'*we; % toeplitz matrix
        Q = R2*psi'*(J'*we*J+wDf)*psi;
%         disp(max(svd(Q-L*J*psi')));
    else
        z = tf('z',Ts);
        xi = (z-1)/(z*Ts); % Joint input shaping....... (F boeren)
%         xi = 1/(Ts*(1-z^-1)); % Rational basis functions... J bolder
        %Compute learning filters efficiently
        JPsi = zeros(N,mpsi);
        for indexBasisFunction = 1 : mpsi
            if basisList(indexBasisFunction) == 1
                Psi(:,indexBasisFunction) = yref;
            elseif basisList(indexBasisFunction) == 2
                Psi(:,indexBasisFunction) = sign(lsim(xi,yref));
            else
                Psi(:,indexBasisFunction) = lsim(xi^(basisList(indexBasisFunction)-2),yref);
            end
            JPsi(:,indexBasisFunction) = lsim(PS,Psi(:,indexBasisFunction));
        end
        R = JPsi.'*we*JPsi+Psi.'*(wf+wDf)*Psi;
        Q = R\(JPsi.'*we*JPsi+Psi.'*wDf*Psi);
        L = R\(JPsi.'*we); % simulatie
    end

    %% trials
    if nargin == 9
        f_jplus1 = zeros(N,1);
        theta_jplus1 = zeros(mpsi,1);
    elseif nargin == 10
        f_jplus1 = Psi*(varargin{10})';
        theta_jplus1 = varargin{10}';
    else
        error('Specifiy 9 or 10 arguments for this function!')
    end
    
    if plotToggle
        t = tin;
        PlotTrialData;
    end
    % Initialize storage variables.
    history.f           = NaN(N,N_trial);
    history.u           = NaN(N,N_trial);
    history.e           = NaN(N,N_trial);
    history.eNorm       = NaN(1,N_trial);
    history.eInfNorm    = NaN(1,N_trial);

    for trial = 1:N_trial
        f_j = f_jplus1;
        theta_j = theta_jplus1;

        f_jsim = timeseries(f_j,tin);
        sim('flexibleBeamSim.slx','SrcWorkspace','current');

        % data acq.
        data = load('data.mat');
        t = data.ans(1,:);
        e_j = data.ans(2,:);
        u_j = data.ans(3,:);
        r = data.ans(4,:);

        % Store trial data.
        history.f(:,trial)          = f_j;
        history.u(:,trial)          = u_j;
        history.e(:,trial)          = e_j;
        history.eNorm(:,trial)      = norm(e_j,2);
        history.eInfNorm(:,trial)   = norm(e_j,Inf);
        if plotToggle
            PlotTrialData;
        end

        theta_jplus1 = Q*theta_j+0.75*L*e_j';
        f_jplus1 = Psi*theta_jplus1;

    end
end

