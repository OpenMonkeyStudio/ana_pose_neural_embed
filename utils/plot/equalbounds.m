function varargout = equalbounds(varargin)
% equalbounds(axType)
% equalbounds(ax,axType)
% lim = equalbounds(...)
%
% set the axis Axtype (eg "xlim") to have equal bounds

%checks
if ishandle(varargin{1})
    ax  = varargin{1};
    varargin(1) = [];
else
    ax = gca;
end
axType = varargin{1};

% set the bounds
lim = get(ax,axType);
if iscell(lim)
    lim = cat(1,lim{:});
end
mx = max(abs(lim(:)));
newlim =[-mx,mx];
set(ax,axType,newlim)

if nargout > 0
    varargout{1} = newlim;
end