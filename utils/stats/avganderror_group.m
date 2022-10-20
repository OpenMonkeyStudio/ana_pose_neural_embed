function [mu,se,iG] = avganderror_group(G,X,avgstr,nboot,dim)
% [mu,se,iG] = avganderror_group(G,X,avgstr)
% [...] = avganderror_group(G,X,avgstr,nboot)
% [...] = avganderror_group(G,X,avgstr,nboot,dim)

% checks
if nargin < 3 || isempty(avgstr)
    avgstr = 'mean';
end
if nargin < 4 || isempty(nboot)
    nboot = 0;
end
if nargin < 5 || isempty(dim)
    dim = 1;
end

% func
dim = 1;
if strcmp(avgstr,'mean')
    f = @(x) nanmean(x,dim);
    s = @(x) nanstd(x,[],dim);
else
    f = @(x) nanmedian(x,dim);
    s = @(x) 1.4826 * mad(x,1,dim);
end

[ug,~,iG] = unique(G);

% prep so that data is along first dimension
doReshape = ndims(X)>2;
if doReshape    
    sz = size(X);
    idx = 1:ndims(X);
    idx = [dim,setxor(idx,dim)];
    X = permute(X,idx);
    X = reshape(X,[sz(idx(1)) prod(sz(idx(2:end)))]);
end

% get rid of columsn of all nan
%badcol = all(isnan(X));
%X(:,badcol) = [];

% get means
if nboot > 1
    mu = grpstats(X, iG, {f});
    se = nan(size(mu));
    for ig=1:numel(ug)
        selg = iG==ug(ig);
        tmp = X(selg,:);
        se(ig,:) = nanstd(bootstrp(nboot,f,tmp));
    end
    %se = grpstats(X, iG,{@(x) nanstd(bootstrp(nboot,f,x))});
else
    mu = grpstats(X, iG, {f});
    se = grpstats(X, iG,{s});
    %se = grpstats(X, iG,{@(x) nanstd(bootstrp(nboot,f,x))});
end

% reshape back
sz2 = size(mu);
if doReshape
    mu = reshape(mu,[sz2(1), sz(idx(2:end))]);
    se = reshape(se,[sz2(1), sz(idx(2:end))]);
end

% make sure it matches group sizes
mx = max(ug);
if size(mu,1) < mx
    tmpmu = nan(mx,sz2(2:end));
    tmpse = nan(mx,sz2(2:end));
    
    idx = ismember(1:mx,ug);
    tmpmu(idx,:,:,:,:,:,:,:) = mu;
    tmpse(idx,:,:,:,:,:,:,:) = se;
    mu = tmpmu;
    se = tmpse;
end

foo=1;