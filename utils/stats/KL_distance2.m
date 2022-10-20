% USAGE: [D] = KL_distance(P, Q)
%
% Computes the Kullback-Leibler (KL) distance - used to infer the amount of
% difference between two distributions

function [D] = KL_distance2(P, Q)

if length(P) ~= length(Q)
    D = 0;
    return
end

ind = find(P ~= 0 & Q~=0);
D = sum(P(ind).*log10(P(ind)./Q(ind)));