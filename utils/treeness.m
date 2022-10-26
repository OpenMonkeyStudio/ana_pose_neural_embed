%function T = treeness(A,clabels_seq,nstate,nseq)
    
% cluster
nclust = 2:max(clabels);

nboot = 1000;
nsmp = size(Y,1);
sel = randperm(nsmp,nboot);
Ytmp = Y(sel,:);

Z = linkage(Ytmp,'ward','euclidean');

clust = [];
for ic=1:numel(nclust)
    c = cluster(Z,'MaxClust',nclust(ic));
    clust(:,ic) = c;
end

return
% backward probabilities
Pf = [];
for iseq=1:nseq
    for istate=1:nstate
        sel = clabels_seq==iseq;
        a = A(sel,:);
        p = accumarray(a(:),1,[nstate 1]);
        p = p + eps;
        p = p ./ sum(p(:));
        Pf(iseq,:) = p;
    end
end

% forward probabilities
Pb = [];
Aseq = repmat(clabels_seq,1,size(A,2));
Aseq = Aseq(:);
Astate = A(:);
for istate=1:nstate
    for iseq=1:nseq
        sel = Astate==istate;
        a = Aseq(sel,:);
        a(isnan(a)) = [];
        p = accumarray(a(:),1,[nseq 1]);
        p = p + eps;
        p = p ./ sum(p(:));
        Pb(:,istate) = p;
    end
end

% treeness
Hf = -sum(sum(Pf .* log(Pf)));
Hb = mean(-sum(Pb .* log(Pb),2));
T = (Hf - Hb) ./ Hf;

