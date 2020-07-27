%PLOTTRIALDATA   Plot trial during simulation/experiment.

%% Initialization
if ~exist('PlotInit','var')
    itplot = figure('NumberTitle','off','Name','Trial data','Units','Normalized','Position',[0.25, 0.1, 0.5, 0.8]);
    
    %% Feedforward.
    ax(1) = subplot(4,1,1);
    hold on;
    pl_fprev_x = plot(t,NaN(N,1),'--','Color',c1);
    pl_f_x = plot(t,NaN(N,1),'Color',c1);
    pl_fprev_y = plot(t,NaN(N,1),'--','Color',c2);
    pl_f_y = plot(t,NaN(N,1),'Color',c2);
    
    hL = legend('Previous iteration x','Current iteration x','Previous iteration y','Current iteration y');    
    newPosition = [0.83 0.9 0.17 0.1];
    newUnits = 'normalized';
    set(hL,'Position', newPosition,'Units', newUnits);
    xlim([0,t(end)]);
    xlabel('t $[s]$');
    ylabel('f $[V]$');
    
    % Title.
    title(ax(1),sprintf('Trial Number %d', N_trial));
    
    %% Control input.
    ax(2) = subplot(4,1,2);
    xlim([0,t(end)]);
    hold on;
    
    pl_uprev_x = plot(t,NaN(N,1),'--','Color',c1);
    pl_u_x = plot(t,NaN(N,1),'Color',c1);
    pl_uprev_y = plot(t,NaN(N,1),'--','Color',c2);
    pl_u_y = plot(t,NaN(N,1),'Color',c2);
    
    
    xlabel('t $[s]$');
    ylabel('u $[V]$');
    
    %% Error.
    ax(3) = subplot(4,1,3);
    hold on;
    
    pl_eprev_x = plot(t,NaN(N,1),'--','Color',c1);
    pl_e_x = plot(t,NaN(N,1),'Color',c1);
    pl_eprev_y = plot(t,NaN(N,1),'--','Color',c2);
    pl_e_y = plot(t,NaN(N,1),'Color',c2);
    
    xlim([0,t(end)]);
    xlabel('t $[s]$');
    ylabel('e $[m]$');
    
    %% Error norm.
    ax(4) = subplot(4,1,4);
    hold on;
    
    pl_eNorm_x = semilogy(0:N_trial-1,NaN(1,N_trial),'--x','Color',c1);
    pl_eNorm_y = semilogy(0:N_trial-1,NaN(1,N_trial),'--x','Color',c2);
    set(ax(4),'XTick',0:N_trial-1);
    xlabel('Trial \#');
    ylabel('$\|e\|_2 [m^2]$');
    if N_trial > 1
        xlim([0,N_trial-1]);
    end
    
    % Link time axes.
    linkaxes(ax(1:3),'x');
    
    % Set init done flag.
    PlotInit = 1;
    
else
    %% Update figure
    
    % Update title.
    title(ax(1),sprintf('Trial %d/%d',trial,N_trial));
    
    % Feedforward.
    set(pl_fprev_x,'YData',get(pl_f_x,'YData'));
    set(pl_f_x,'YData',history.fx(:,trial));
    set(pl_fprev_y,'YData',get(pl_f_y,'YData'));
    set(pl_f_y,'YData',history.fy(:,trial));
    
    % Control input.
    set(pl_uprev_x,'YData',get(pl_u_x,'YData'));
    set(pl_u_x,'YData',u_j_x);
    set(pl_uprev_y,'YData',get(pl_u_y,'YData'));
    set(pl_u_y,'YData',u_j_y);
    
    % Error.
    set(pl_eprev_x,'YData',get(pl_e_x,'YData'));
    set(pl_e_x,'YData',e_j_x);
    set(pl_eprev_y,'YData',get(pl_e_y,'YData'));
    set(pl_e_y,'YData',e_j_y);
    
    % Error norm.
    set(pl_eNorm_x,'YData',history.eNormx,'Color',c1);
    set(pl_eNorm_y,'YData',history.eNormy,'Color',c2);
    
end

% Flush drawing.
drawnow;