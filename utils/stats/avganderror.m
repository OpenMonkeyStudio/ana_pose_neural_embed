function [mu,se] = avganderror(X,avgstr,dim,getBootstrappedSE,nboots)
% [mu,se] = avganderror(X,avgstr)
% [mu,se] = avganderror(X,avgstr,dim)
% [mu,se] = avganderror(X,avgstr,dim,getBootstrappedSE)
% [mu,se] = avganderror(X,avgstr,dim,getBootstrappedSE,nboots)
%
% convenience function to compute the central tendency and associated standard
% deviation. 
%
% X: ND array
% avgstr: 'mean','median' (omits nans)
% dim: dimension to opertae over
% getBootstrappedSE: whether to calcultae the standard error usinng
% bootstrapping (eg for median)
% nboots: number of bootstrap samples

%inputs
if nargin < 3 || isempty(dim)
    dim = 1;
    if isrow(X); X = X'; end
end

if nargin < 4; getBootstrappedSE = 0; end
if getBootstrappedSE && nargin<5; nboots = 2000; end

%calculate
if strcmp(avgstr,'median')
    mu = nanmedian(X,dim);
    %se = 1.253 * nanstd(X,[],dim) ./ sqrt(size(X,dim)); %shouldnt use this...
    se = 1.4826 * mad(X,1,dim) ./ sqrt(size(X,dim));

    func = @nanmedian;
elseif strcmp(avgstr,'mean')
    mu = nanmean(X,dim);
    se = nanstd(X,[],dim) ./ sqrt(size(X,dim));
    func = @nanmean;
elseif strcmp(avgstr,'prop')
    mu = sum(X>0) ./ size(X,1);
    se = nan(size(mu));
    func = @(x) sum(x>0) ./ size(x,1); 
else
    error('huh?')
end

%bootstrap SE?
if getBootstrappedSE
    % prep
    sz = size(X);
    idx = [dim,setxor(1:ndims(X),dim)];
    X = permute(X,idx);
    X = reshape(X,[sz(dim), prod(sz)./sz(dim)]);
    
    % bootstrap
    b = bootstrp(nboots,func,X,1);
    se = nanstd(b,[],1);
    
    % finalize shape
    sz2 = sz(idx);
    sz2(1) = 1;
    se = reshape(se,sz2);

    foo=1;
end
