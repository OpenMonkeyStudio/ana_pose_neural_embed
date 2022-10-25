function [nr,nc] = subplot_ratio(ntot,ratio)
% [nr,nc] = subplot_ratio(ntot)
% [nr,nc] = subplot_ratio(ntot,ratio)
%
% returns number of rows "nr" and columns "nc" that maintains some "ratio"
%
% ratio=ncol/nrow. if ratio not supplied, uses ratio of screen size

if nargin <2
    sz = get(0,'screensize');
    ratio = sz(3)/sz(4); %col/row
end

nr = ceil(sqrt( ntot ./ ratio ));
nc = ceil( ntot./nr );

foo=1;