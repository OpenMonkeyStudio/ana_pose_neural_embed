function out = get_graph_cluster(C,idat,anadir)
% out = get_graph_cluster(C,idat,anadir)

%settings
nrand = 10;
trans_lags = [1:1:49, 50:5:100, 110:10:200, 300:100:1000];

% paths
[parentdir,jsondir,pyenvpath,rpath,binpath,codepath,ephyspath] = set_pose_paths(0);

mpath = [anadir '/modularity_test'];
if ~exist(mpath); mkdir(mpath); end

figdir = [anadir '/Figures'];
if ~exist(figdir); mkdir(figdir); end

% useful
nstate = max(C);

istate = find(diff(C)~=0);
C_state = C(istate);

idat_state = idat(istate);

% save info
sname = [mpath '/modInfo.mat'];
save(sname,'trans_lags','nrand')

fprintf('graph clustering:\n \t %s\n',anadir)
tic

%% states that arent in datasets
ignoredStates = {};
for id=1:max(idat)
    sel = idat_state==id;
    c = C_state(sel);
    n = accumarray(c,1,[nstate 1]);
    bad = find(n==0);
    good = find(n~=0);
    idx = [good; bad];

    ignoredStates{id,1} = idx;
    ignoredStates{id,2} = bad;
    ignoredStates{id,3} = good;
end

sname = [mpath '/ignoredStates.mat'];
save(sname,'ignoredStates')
    

%% ------------------------------------------------------------
% observed
fprintf('getting all OBSERVED transitions')
nsmp = numel(trans_lags)*max(idat);
PO = cell(1,nsmp);
ii = 0;
for id=1:max(idat)
    for ilag = 1:numel(trans_lags)
        dotdotdot(ii,0.1,nsmp)
        sel = idat(istate)==id;
        c = C_state(sel);
        thisLag = trans_lags(ilag);

        po = cond_trans_prob(c,thisLag,nstate,1);

        % deal with non-existant states in this data
        good = ignoredStates{id,3};
        po = po(good,good);

        % store
        ii = ii+1;
        PO{ii} = po;
    end
end
fprintf('\n')

% call clustering
dat = [];
dat.po = PO;
dat.dim = 'lag-idat';
dat.func = 'fit_paris';
dat.make_cuts = 1;

opts = [];
opts.verbose = 1;
opts.clean_files = 0;
opts.tmpname = [mpath '/po_obs_hier'];
opts.func = [codepath '/utils/python/call_transition_cluster.py'];
opts.pyenvpath = pyenvpath;

out_obs = sendToPython(dat,opts);

sz = [numel(trans_lags) max(idat)];
out_obs = reformat_graph_cluster(out_obs,sz,ignoredStates);

% add transition matrices, for later plotting
po2 = reshape(PO,sz);
out_obs.po = po2;

%% ------------------------------------------------------------
% random
fprintf('getting all RANDOM transitions')
nsmp = numel(trans_lags)*max(idat)*nrand;
PO = cell(1,nsmp);
ii = 0;

for ir=1:nrand
    for id=1:max(idat)
        sel = idat(istate)==id;
        c = C_state(sel);
        c = c(randperm(numel(c)));

        for ilag = 1:numel(trans_lags)
            dotdotdot(ii,0.1,nsmp)

            thisLag = trans_lags(ilag);

            po = cond_trans_prob(c,thisLag,nstate,1);

            % deal with non-existant states in this data
            good = ignoredStates{id,3};
            po = po(good,good);

            % store
            ii = ii+1;
            PO{ii} = po;
        end
    end
end
fprintf('\n')

% call clustering
dat = [];
dat.po = PO;
dat.dim = 'lag-idat';
dat.func = 'fit_paris';
dat.make_cuts = 1;

opts = [];
opts.verbose = 1;
opts.clean_files = 0;
opts.tmpname = [mpath '/po_rand_hier'];
opts.func = [codepath '/utils/python/call_transition_cluster.py'];
opts.pyenvpath = pyenvpath;

out_rand = sendToPython(dat,opts);

sz = [numel(trans_lags) max(idat) nrand];
out_rand = reformat_graph_cluster(out_rand,sz,ignoredStates);

%% finalize and save
out = [];
out.obs = out_obs;
out.rand = out_rand;
out.ignoredStates = ignoredStates;
out.trans_lags = trans_lags;
out.Cutoffs = 2:nstate-1;

fprintf('\t saving: %s\n',sname)
sname = [mpath '/modularity_train.mat'];
save(sname,'-v7.3','-struct','out')

toc


%% MISC

function out = reformat_output(in,sz,ignoredStates)
    try, in.dendrogram = in.dendogram; in = rmfield(in,'dendogram'); end
    
    % prep
    nstate = numel(ignoredStates{1});
    ndat = size(ignoredStates,1);
    mx_ncut = cellfun(@(x) max(x),in.ncut);
    
    % reformat each cell
    out = in;
    
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
        
%         % number of cuts
%         m = out.ncut{ii};
%         tmp = nan(1,nstate);
%         tmp(1:numel(m)) = m;
%         out.ncut{ii} = tmp;
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
    
    % maximum modularity
    [mx,imx] = max(out.modularity,[],3);
    out.max_modularity = mx;
    out.imax_modularity = imx;
    
    % finalize
    out.dim = 'lag-dataset-ncut-state';
    %out.ignoredStates = ignoredStates;

    foo=1;