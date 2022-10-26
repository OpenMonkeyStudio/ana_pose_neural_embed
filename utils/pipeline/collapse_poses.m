function [X2,c2,istate] = collapse_poses(X,c,winlim,istate)
% [X2,c2,istate] = collapse_poses([],c,winlim)
% [X2,c2,istate] = collapse_poses(X,c,winlim)
% [X2,c2,istate] = collapse_poses(X,c,winlim,istate)

% checks
minWin = winlim(1);
maxWin = winlim(2);

if nargin < 4 || isempty(istate)
    getStateIdx = 1;
else
    getStateIdx = 0;
end

collapseData = ~isempty(X);

if collapseData
    doTranspose = isrow(X);
    if doTranspose
        X = X'; 
    end
end

% split into "trials"
if getStateIdx
    istate = find( diff(c)~=0 );
    istate = [0; istate; numel(c)-1];

    dt = diff(istate);
    tooShort = find(dt < minWin)+1;
    istate(tooShort) = [];

    dt = diff(istate);
    tooLong = find(dt > maxWin);

    tmps = istate;
    for it=1:numel(tooLong)
        it2 = tooLong(it);
        n = ceil( dt(it2) ./ maxWin );
        idx = floor(linspace(0,dt(it2),n+1))';
        idx(end)=[];

        pre = istate(1:it2-1);
        mid = idx+istate(it2);
        post = istate(it2+1:end);
        tmps = [pre;mid;post];
    end
    istate = tmps;
end

% compile
if collapseData
    X2 = nan(numel(istate)-1,size(X,2));
    for is=1:numel(istate)-1
        st = istate(is)+1;
        fn = istate(is+1);

        tmp = X(st:fn,:);
        X2(is,:) = nanmean(tmp);
    end
end

if ~isempty(c)
    c2 = c(istate(1:end-1)+1);
else
    c2 = [];
end

% get rid of nans
bad = find( any(isnan(X2),2) );
X2(bad,:) = [];
c2(bad) = [];
istate(bad+1) = [];

% finish
if collapseData
    if doTranspose
        X2 = X2';
    end
else
    X2 = [];
end