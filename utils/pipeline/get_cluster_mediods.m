function [M,uclust]=get_cluster_mediods(X,labels)
% [M,uclust]=get_cluster_mediods(X,labels)

% unique clusters
uclust = unique(labels(~isnan(labels)));
nclust = numel(uclust);

% mediods of each
M = nan(nclust,size(X,2));
for ic=1:nclust
    sel = labels==uclust(ic);
    tmp = X(sel,:);
    M(ic,:) = nanmedian(tmp);
end
