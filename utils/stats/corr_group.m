function [r,p,ugroup]=corr_group(G,x,y,varargin)
% [r,p,ugroup]=corr_group(G,x,y)
% [r,p,ugroup]=corr_group(G,x,y,varargin)

[ug,~,ig] = unique(G);

r = [];
p = [];
for ii=1:numel(ug)
    sel = ig==ug(ii);
    xx = x(sel);
    yy = y(sel);
    
    [r(ii) p(ii)] = corr(xx,yy,varargin{:});
end

ugroup = ug;