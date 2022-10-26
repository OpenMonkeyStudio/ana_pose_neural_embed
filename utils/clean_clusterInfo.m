function cluster_train_out = clean_clusterInfo(cluster_train_in,stateMap)
% cluster_train_out = clean_clusterInfo(cluster_train_in,stateMap)

cluster_train_out = cluster_train_in;

% remap
oldval = stateMap(1,:);
newval = stateMap(2,:);

cluster_train_out.Ld = changem(cluster_train_out.Ld,newval,oldval);
cluster_train_out.clabels = changem(cluster_train_out.clabels,newval,oldval);

% new boundaries
if 0
    dens = cluster_train_out.outClust.dens2;
    %dens(cluster_train_out.Ld==0) = 0;
    %dens(isnan(cluster_train_out.Ld) | cluster_train_out.Ld==0) = 0;
    L = watershed(-dens);
    L = uint8(L); % BV
    BW = dens > min(dens(:));
    L2 = L .* uint8(BW);

    Lin = ~bwperim(L2) .* BW .* ~L2;
    Lbnds = bwperim(BW) | Lin;
    Lbnds(isnan(cluster_train_out.Ld) | cluster_train_out.Ld==0) = 0;
    cluster_train_out.Lbnds = Lbnds;
else
    % get rid of missing clusters
    Lbnds = cluster_train_out.Lbnds;
    Lbnds(isnan(cluster_train_out.Ld) | cluster_train_out.Ld==0) = 0;
    missingBorders = bwperim(cluster_train_out.Ld>0);
    missingBorders = missingBorders==1 & Lbnds==0;
    
    % clean
    Lbnds(missingBorders==1) = 1;
    Lbnds = bwmorph(Lbnds,'fill'); 
    Lbnds = bwmorph(Lbnds,'thin');
    
    cluster_train_out.Lbnds = Lbnds;
end

% update
bad = cluster_train_out.Ld==0;
cluster_train_out.outClust.dens2(bad) = 0; 
