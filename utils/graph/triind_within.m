function iwith = triind_within(nx,c,ic)
% iwith = triind_within(nx,c,ic)
%
% works with output of pdist. returns logical matrix for all pairs in the
% output (vector form) that are within the same cluster


% find all samples of this cluster
selc = find(c==ic);

% for speed
ndist = nchoosek(nx,2);
iwith = false(1,ndist);
for is=numel(selc):-1:2
    col = selc(selc<selc(is));
    row = ones(size(col))*selc(is);
    ii = sub2triind([nx 1],row,col);
    iwith(ii) = true;
end
