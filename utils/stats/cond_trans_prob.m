function [po,s] = cond_trans_prob(c_state,lag,nstate,w,getCellMean)
% po = cond_trans_prob(c_state,lag,nstate,w,getCellMean)
% [po,pos] = cond_trans_prob(c_state,lag,nstate,w,getCellMean)
%
% warning: when weighted, columsn dont sum to 1

% check
if nargin < 3 || isempty(nstate)
    nstate = max(c_state);
end
if nargin < 4 || isempty(w)
    w = 1;
end
if nargin < 5 || isempty(getCellMean)
    getCellMean = 0;
end

% indices, accounting for lag
%fn = lag+1:lag:numel(c_state);
%st = 1:lag:numel(c_state)-lag;
fn = lag+1:1:numel(c_state);
st = 1:1:numel(c_state)-lag;
if numel(w) > 1 && (numel(w) ~= numel(st))
    error('size of weights not right')
end

% get matrix
s = accumarray(c_state(st),w,[nstate 1])';
po = accumarray([c_state(fn),c_state(st)],w,[nstate nstate]);
if getCellMean
    tmps = accumarray([c_state(fn),c_state(st)],1,[nstate nstate]);
    po = po ./ tmps;
else
    po = po ./ s;
end
po(:,s==0) = 0;