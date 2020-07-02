%PLOTTRIALDATA   Plot trial during simulation/experiment.

%% Initialization
if ~exist('PlotInit','var')
    itplot = figure('NumberTitle','off','Name','Trial data','Units','Normalized','Position',[0.25, 0.1, 0.5, 0.8]);
    
    % Feedforward.
    ax(1) = subplot(4,1,1);
    hold on;
    pl_fprev = plot(t,NaN(N,1),'k-');
    pl_f = plot(t,NaN(N,1),'r--');
%     sref1 = plot(t,NaN(N,1),'b:','Linewidth',1.5);
%     aref1 = plot(t,NaN(N,1),'c:','Linewidth',1.5);
%     hL = legend('Previous iteration','Current iteration','Scaled Snap','Scaled Acceleration');
    hL = legend('Previous iteration','Current iteration');    
    newPosition = [0.83 0.9 0.17 0.1];
    newUnits = 'normalized';
    set(hL,'Position', newPosition,'Units', newUnits);
    xlim([0,t(end)]);
    xlabel('t $[s]$');
    ylabel('f $[V]$');
    
    % Title.
    title(ax(1),sprintf('Trial Number %d', N_trial));
    
    % Control input.
    ax(2) = subplot(4,1,2);
    xlim([0,t(end)]);
    hold on;
    pl_uprev = plot(t,NaN(N,1),'k-');
    pl_u = plot(t,NaN(N,1),'r--');
    xlabel('t $[s]$');
    ylabel('u $[V]$');
    
    % Error.
    ax(3) = subplot(4,1,3);
    hold on;
    pl_eprev = plot(t,NaN(N,1),'k-');
    pl_e = plot(t,NaN(N,1),'r--');
%     sref3 = plot(t,NaN(N,1),'b:','Linewidth',1.5);
%     aref3 = plot(t,NaN(N,1),'c:','Linewidth',1.5);
    xlim([0,t(end)]);
    xlabel('t $[s]$');
    ylabel('e $[m]$');
    
    % Error norm.
    ax(4) = subplot(4,1,4);
    pl_eNorm = plot(0:N_trial-1,NaN(1,N_trial),'r--x');
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
    set(pl_fprev,'YData',get(pl_f,'YData'));
    set(pl_f,'YData',f_j);
%     set(sref1,'YData',d/max(d)*max(f_j));
%     set(aref1,'YData',a/max(a)*max(f_j));
    
    % Control input.
    set(pl_uprev,'YData',get(pl_u,'YData'));
    set(pl_u,'YData',u_j);
    
    % Error.
    set(pl_eprev,'YData',get(pl_e,'YData'));
    set(pl_e,'YData',e_j);
%     set(sref3,'YData',d/max(d)*max(e_j));
%     set(aref3,'YData',a/max(a)*max(e_j));
    
    % Error norm.
    set(pl_eNorm,'YData',history.eNorm);
    
end

% Flush drawing.
drawnow;