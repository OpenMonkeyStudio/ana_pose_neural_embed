function k = kldiv2(x,p1,p2,varargin)
% k = kldiv2(x,p1,p2,varargin)
%
% wrapper for kldiv, for >1D data

% no zeros
p1 = p1 + eps;
p1 = p1 ./ sum(p1(:));

p2 = p2 + eps;
p2 = p2 ./ sum(p2(:));

% KL
k = kldiv(x,p1,p2,varargin{:});