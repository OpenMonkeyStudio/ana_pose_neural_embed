function out = reformat_output(in,sz,ignoredStates)
% out = reformat_output(in,sz,ignoredStates)

    try, in.dendrogram = in.dendogram; in = rmfield(in,'dendogram'); end
    
    % prep
    nstate = numel(ignoredStates{1});
    ndat = size(ignoredStates,1);
    
    bad = cellfun(@isempty,in.ncut);
    in.tree_sampling_divergence(bad) = nan;
    in.dasgupta_score(bad) = nan;
    in.modularity(bad) = {nan};
    in.ncut(bad) = {nan};
    
    mx_ncut = cellfun(@(x) max(double(x)),in.ncut);
    
    out = rmfield(in,'ncut');

    % reformat each cell
    
    for ii=1:numel(out.dendrogram)
        id = mod(ii-1,prod(sz(1:2)))+1;
        id = ceil(id/sz(1));
        
        % dendogram
        z = in.dendrogram{ii};
        z = z(:,1:3);
        z(:,1:2) = z(:,1:2)+1;    
        
        out.dendrogram{ii} = z;
        
        % labels
        good = ignoredStates{id,3};
        
        oldlab = out.labels_cut{ii};
        newlab = nan(nstate-2,nstate);
        newlab(1:size(oldlab,1),good) = oldlab+1;
        
        out.labels_cut{ii} = newlab;
        
        % modularity
        m = out.modularity{ii};
        tmp = nan(nstate-2,1);
        tmp(1:numel(m)) = m;
        out.modularity{ii} = tmp;
    end

    % reshape for ease later
    ff = fields(out);
    for ii=1:numel(ff)
        f = ff{ii};
        out.(f) = reshape(out.(f),sz);
    end
    
    % concatenate for ease
    if numel(sz)==2 % obs
        idim = [3 4 1 2];
    elseif numel(sz)==3 % rand
        idim = [3 4 1 2 5];
    end
    
    tmp = out.modularity;
    sz1 = size(tmp);
    sz2 = size(tmp{1});
    tmp = cat(3,tmp{:});
    tmp = reshape(tmp,[sz2 sz1]);
    tmp = permute(tmp,idim);
    out.modularity = tmp;
    
    tmp = out.labels_cut;
    sz1 = size(tmp);
    sz2 = size(tmp{1});
    tmp = cat(3,tmp{:});
    tmp = reshape(tmp,[sz2 sz1]);
    tmp = permute(tmp,idim);
    out.labels_cut = tmp;
    
    % finalize
    out.dim = 'lag-dataset-ncut-state-(rand?)';
    %out.ignoredStates = ignoredStates;

    foo=1;