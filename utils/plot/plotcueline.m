function varargout = plotcueline(varargin)
% plotcueline(axplot,axval)
% plotcueline(axplot,axval,{optional inputs to plotting function})
% plotcueline(hax,...)
% h = plotcueline(...)
%
% plots a straight line along the x or yaxis at the value axval into the axes "haxall". if
% haxall is empty, plots into all axes. if haxall is not provided, plots
% into current axis

%determine axis
if ishandle(varargin{1})
    haxall = varargin{1};
    varargin(1) = [];
elseif isempty(varargin{1})
    haxall = findobj(gcf,'type','axes'); 
    haxall = haxall(:);
    varargin(1) = [];

else
    haxall = gca;
end

axplot = varargin{1};
axval = varargin{2};

% set default optional inputs to plotting function if theyre not supplied
if numel(varargin)>2
    plotargs = varargin(3:end);
else
    plotargs{1} = 'k-';
    plotargs{2} = 'linewidth';
    plotargs{3} = 1;
end

if strncmpi(axplot,'x',1)
    plotX = 1;
elseif strncmpi(axplot,'y',1)
    plotX = 0;
else
    error('unrecognized axis')
end

%loop over each axis
hl = [];
for n=1:numel(haxall)
    hax = haxall(n);
    
    origx = get(hax,'xlim');
    origy = get(hax,'ylim');
    
    %plot each line
    for iplot=1:numel(axval)
        if plotX
            yval = get(hax,'ylim');
            xval = axval(iplot) .* ones(size(yval));
        else 
            xval = get(hax,'xlim');
            yval = axval(iplot) .* ones(size(xval));
        end

        %plot line
        hold(hax, 'all')
        hl(n,iplot) = plot(hax,xval,yval,plotargs{:});
    end
    
    %adjust the limits
    if 0
        if plotX
            set(hax,'ylim',origy);
            mn = min( min(axval), min(origx) );
            mx = max( max(axval), max(origx) );
            set(hax,'xlim',[mn mx])
        else
            set(hax,'xlim',origx);
            mn = min( min(axval), min(origy) );
            mx = max( max(axval), max(origy) );
            set(hax,'ylim',[mn mx])
        end
    end

end

if nargout>0
    varargout{1} = hl;
end