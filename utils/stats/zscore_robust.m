function [z,mu,sd] = zscore_robust(x,dim,allFlag)
% [z] = zscore_robust(x)
% [z] = zscore_robust(x,dim)
% [z] = zscore_robust(x,[],allFlag)
% [z,mu,sd] = zscore_robust(...)

% checks
if nargin < 2 || isempty(dim)
    sz = size(x);
    dim = find(sz>1,1);
end
if nargin < 3 || isempty(allFlag)
    allFlag = '';
end

% calculate
if strcmp(allFlag,'all')
    mu = median(x(:));
    sd = mad(x(:),1) * 1.4826;
else
    mu = median(x,dim);
    sd = mad(x,1,dim) * 1.4826;
end
z = (x-mu)./sd;

% finish
bad = x==mu;
z(bad) = 0;


