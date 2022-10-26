%function T = treeness(A,clabels_seq,nstate,nseq)

% useful
nclust = max(clabels);
nboot = 1000;
nsmp = size(Y,1);


% for each cluster number
%
Pf = [];
Pb = [];
for ic=1:nclust(ic)-1
    nc = nclust(ic);
    
    % reample the data
    nc1 = nclust(ic);
    nc2 = nclust(ic+1);

    sel = randperm(nsmp,nboot);
    Ytmp1 = Y(sel,:);
    c1 = clabels(sel);
    sel = randperm(nsmp,nboot);
    Ytmp2 = Y(sel,:);
    c2 = clabels(sel);

    % create partitions
    Z1 = linkage(Ytmp1,'ward','euclidean');
    Z2 = linkage(Ytmp2,'ward','euclidean');
    t1 = cluster(Z1,'MaxClust',nc1);
    t2 = cluster(Z1,'MaxClust',nc2);

    % backwards prob
    for is1=1:nclust
        for is2=1:nclust
            
        end
    end
    
    for it1=1:nc1
        for it2=1:nc2
            sel = t1==it1;
            n = accumarray(
        end
    end
    p1 = accumarray([c1 c2],1);

    % forward prob
    foo=1;
end
%}


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

