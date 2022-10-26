function [iwith1,iwith2,ndist,nw1,nw2,nb] = triind_twocluster(n,n1,n2)

% compute within-cluster indices
ndist = nchoosek(n,2);
nw1 = nchoosek(n1,2);
nw2 = nchoosek(n2,2);
nb = ndist-nw1-nw2;

iwith1 = false(1,ndist);
s=-n+1;
for is=1:n1-1
    s = s+n-is+1;
    f = s+(n1-is)-1;
    iwith1(s:f) = 1;
end

s = nw1+nb+1;
iwith2 = false(1,ndist);
iwith2(s:end) = 1;