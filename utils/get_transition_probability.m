function [P,N] = get_transition_probability(states,step,ustates)
% [P,N] = get_transition_probability(states)
% [P,N] = get_transition_probability(states,step,ustates)

% check
if nargin < 2 || isempty(step)
    step = 1;
end

[tmp,~,ic] = unique(states);
tmp(isnan(tmp)) = [];
if nargin <3
    ustates = tmp;
end

% prep
nu = numel(unique(ustates));

xedge = [[1:nu], nu+1];
yedge = [[1:nu], nu+1];

x = ic(1:end-step);
y = ic(step+1:end);

% count transitions
N = histcounts2(x,y,xedge,yedge);
P = N ./ sum(N(:));

foo=1;

