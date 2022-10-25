function varargout = axisequal(ax)
% axisequal()
% axisequal(ax)
%
% set the x and y axis limits to be the same

if nargin<1
    ax  = gca;
end

for ix=1:numel(ax)
    xlim = get(ax,'xlim');
    ylim = get(ax,'ylim');
    lim = [min([xlim,ylim]),max([xlim,ylim])];
    set(gca,'xlim',lim,'ylim',lim)
end

%output
if nargout>0
    varargout{1} = ax;
end