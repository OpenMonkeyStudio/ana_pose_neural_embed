function k = kldiv_nd(p1,p2,varargin)
% k = kldiv_nd(p1,p2,args)
% 
% assums equallly spaced bins for x
%
% see also: kldiv

p1(isnan(p1)) = 0;
p2(isnan(p2)) = 0;
x = [1:numel(p1)]';

% no zeros
p1 = p1(:) + eps;
p1 = p1 ./ sum(p1);

p2 = p2(:) + eps;
p2 = p2 ./ sum(p2);

% KL
k = kldiv(x,p1,p2,varargin{:});