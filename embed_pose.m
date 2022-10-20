function embed_pose(ecfg)

[parentdir,~,pyenvpath,~,binpath,codepath] = set_pose_paths(0);

% checks
ecfg = checkfield(ecfg,'anadir','needit');
ecfg = checkfield(ecfg,'getFeatures','needit');
ecfg = checkfield(ecfg,'train','needit');
ecfg = checkfield(ecfg,'datalog_path','needit');


% flags
getFeatures = 1;
    calcFeatures = 0;
        useOwnFirstData = 0;
    loadFeatures = 0;
    loadNormFeatures = 1;
    
prepTraining = 1;
runEmbedding = 1;
doRembed = 1;

plotEmbedding = 1;
makeExampleVideos = 0;

% settings
monk = 'wo';

nparallel = 15;

doTrim = 0;

% first dataset?
if useOwnFirstData
    if strcmp(monk,'yo')
        baseDataset = 'yo_2021-02-25_01_enviro';
    else
        baseDataset = 'wo_2021-12-08_01_enviro';
    end
else
    baseDataset = 'other';
end

% egocentric features
theseFeatures = {
    'limb4_segmentlength_wvtpca',                
    'limb2_pca_wvtpca',
    'limb3_segmentlength_smooth'
    'limb2_pca'
    };
            
 % world features
theseFeatures2 = {                
    'limb2_cart-y_speed_interstd_smooth';
    'limb2_cart-xz_speed_interstd_smooth';

    'perpendicularity_smooth',
    'height_smooth',

    'heightVelocity_smooth'
    'comSpeed_smooth',
    'groundSpeed_smooth',
    'heightSpeed_smooth',
    };   


% paths
anadir = ecfg.anadir;
datadir = [fileparts(anadir) '/Data_proc_13joint'];

featdir = [anadir '/X_feat'];
if ~exist(featdir); mkdir(featdir); end
infopath = [anadir '/info.mat'];

% deffine datasets
s = ecfg.datalog_path;
%s = [get_code_path() '/bhv_cluster/data_log_yoda_ephys.xlsx'];
%s = [get_code_path() '/bhv_cluster/data_log_woodstock_ephys.xlsx'];

taskInfo = readtable(s);
names = cellfun(@(x) [x '_proc.mat'], taskInfo.name,'un',0);
datasets = struct('name',names,'folder',repmat({[datadir '/data_ground']},size(taskInfo,1),1));

%datasets = datasets(8:end);

START = tic;

%% for each dataset, load and build features
if getFeatures
    
    % run the first dataset to get all feature params
    if ecfg.train
        firstDataset = find(contains({datasets.name},baseDataset));
        if numel(firstDataset)~=1; error('unrecognized dataset'); end
            
        name = datasets(firstDataset).name;
        [out,procInfo] = run_proc(name,datadir,featdir,theseFeatures,theseFeatures2,procInfo);

        % save it
        tmp = [];
        tmp.procInfo = procInfo;
        tmp.datasets = datasets;
        tmp.theseFeatures = theseFeatures;
        tmp.theseFeatures2 = theseFeatures2;
        tmp.feat_labels = out.feat_labels;
        tmp.ifeat = out.ifeat;
        tmp.baseDataset = baseDataset;

        sname = [featdir '/procInfo.mat'];
        parsave(sname,tmp)
    else
        tmp = load([featdir '/procInfo_other.mat']);
        procInfo = tmp.procInfo;
        firstDataset = [];

        tmp.datasets = datasets;

        sname = [featdir '/procInfo.mat'];
        parsave(sname,tmp)
    end
    
    
    % now loop over the rest
    if nparallel > 1 && isempty(gcp('nocreate'))
        myPool = parpool(nparallel);
    end

    tic
    theseSets = setxor(1:numel(datasets),firstDataset);
    parfor id1 = 1:numel(theseSets)
        id = theseSets(id1);
        name = datasets(id).name;
        fprintf('%g: %s\n',id,name)

        [tmppout,~] = run_proc(name,datadir,featdir,theseFeatures,theseFeatures2,procInfo);
    end
    toc
        
        
    
    if calcFeatures
        procInfo = {[],[]};

        % run the firsts dataset to get the processing info
        if useOwnFirstData
            firstDataset = find(contains({datasets.name},baseDataset));
            if numel(firstDataset)~=1; error('unrecognized dataset'); end
            
            name = datasets(firstDataset).name;
            [out,procInfo] = run_proc(name,datadir,featdir,theseFeatures,theseFeatures2,procInfo);

            % save it
            tmp = [];
            tmp.procInfo = procInfo;
            tmp.datasets = datasets;
            tmp.theseFeatures = theseFeatures;
            tmp.theseFeatures2 = theseFeatures2;
            tmp.feat_labels = out.feat_labels;
            tmp.ifeat = out.ifeat;
            tmp.baseDataset = baseDataset;

            sname = [featdir '/procInfo.mat'];
            parsave(sname,tmp)
        else
            tmp = load([featdir '/procInfo_other.mat']);
            procInfo = tmp.procInfo;
            firstDataset = [];

            tmp.datasets = datasets;

            sname = [featdir '/procInfo.mat'];
            parsave(sname,tmp)
        end
        
        % now loop over the rest
        if nparallel > 1 && isempty(gcp('nocreate'))
            myPool = parpool(nparallel);
        end
        
        tic
        theseSets = setxor(1:numel(datasets),firstDataset);
        parfor id1 = 1:numel(theseSets)
            id = theseSets(id1);
            name = datasets(id).name;
            fprintf('%g: %s\n',id,name)

            [tmppout,~] = run_proc(name,datadir,featdir,theseFeatures,theseFeatures2,procInfo);
        end
        toc
        
            
        % save info
        save(infopath,'datasets','theseFeatures','theseFeatures')
    else
        fprintf('loading training info... \n')
        load(infopath)
    end

    
    % load the features back
    if (loadFeatures || calcFeatures) && ~loadNormFeatures
        fprintf('loading back all ORIG features...\n')
        
        tic
        
        [X_feat,tmpdat] = load_features(featdir,'feat',datasets);
        evals(tmpdat); % put into environment
        
        % save
        tmp = tmpdat;
        fprintf('saving info...\n')
        save(infopath,'-append','-struct','tmp')
        clear tmp tmpdat
        
        toc
    end
    
    
    % normalize them
    featdir2 = [featdir '_norm'];
    if ~exist(featdir2); mkdir(featdir2); end
            
        
    if loadNormFeatures
        fprintf('loading back all NORM features...\n')
        [X_feat,tmpdat] = load_features(featdir2,'feat',datasets);
    else
        % normalize
        fprintf('normalizing...\n')
        
        tic
        if 0 % independent norm
            if 1
                X_feat = zscore_robust(X_feat);
            elseif 0
                X_feat = zscore(X_feat);
            else
                X_feat = X_feat - nanmean(X_feat);
            end
        else % set norm
            Xtmp = X_feat;
            for jj=1:numel(ifeat)
                tmp = X_feat(:,ifeat{jj});
                mu = nanmedian([tmp(:)]);
                se = mad([tmp(:)],1,1)*1.4826;
                tmp = (tmp-mu)./se;

                X_feat(:,ifeat{jj}) = tmp;
            end
            clear Xtmp
        end
        toc

        % resave
        tic
        fprintf('resaving each one')
        for id=1:numel(datasets)
            name = datasets(id).name;
            %fprintf('%g: %s\n',id,name)
            fprintf('%g,',id)

            name_in = [featdir '/' name(1:end-4) '_feat.mat'];
            tmp_in = load(name_in);    
            
            % resave
            x = X_feat(idat==id,:);
            
            tmp_out = tmp_in;
            tmp_out.X_feat = x;
            
            name_out = [featdir2 '/' name(1:end-4) '_feat.mat'];
            parsave(name_out,tmp_out)
        end
        fprintf('\n')
        toc
    end
end


%% select training samples
if prepTraining

    % prep
    V = nan(size(frame));
    for id=1:max(idat)
        sel = idat==id;
        tmpf = frame(sel);
        tmpc = com(sel,:);

        dt = diff(tmpf/30);
        d = diff(tmpc,[],1);
        v = sqrt(sum(d.^2,2)) ./ dt;
        v = [0; smooth(v,5)];
        V(sel) = v;
    end

    H = com(:,2);

    % init indices
    idx = 1:6:numel(frame);

    % oversample rare events 
    if 1
        tmpfs = 1;

        %samples with high speed
        selv = V > 2;
        [st,fn] = find_borders(selv);
        tooShort = (fn-st)<2; st(tooShort) = []; fn(tooShort) = [];
        st = max(st - 2,1); fn = min(fn + 2,numel(v));
        for is=1:numel(st); selv(st(is):tmpfs:fn(is)) = 1; end

        % on the wall
        selh = H > 3;
        [st,fn] = find_borders(selh);
        tooShort = (fn-st)<2; st(tooShort) = []; fn(tooShort) = [];
        st = max(st - 2,1); fn = min(fn + 2,numel(v));
        for is=1:numel(st); selh(st(is):tmpfs:fn(is)) = 1; end

        % final 
        selnew = selv | selh;
        idx = [idx, find(selnew)'];
    end
    
    % final
    idx_train = unique(idx);
    X_feat_train = X_feat(idx_train,:);
    idat_train = idat(idx_train);
    frame_train = frame(idx_train);

    % save
    save(infopath,'-append','idx_train')
end


%% run embedding
if runEmbedding

    % fit
    cfg = [];
    cfg.anadir = anadir;
    cfg.nparallel = 0;
    cfg.fs = 30;
    cfg.presave = 0;
    cfg.group = idat_train;
    cfg.train = 1;
    cfg.pca_indiv = 0;
    cfg.pca_global = 0;
    cfg.group_ica = 0;
    cfg.wvt = 0;
    cfg.wvt_pca = 0;
    cfg.umap = 1;
        cfg.umap_class = 'umap';
        %cfg.umap_cfg.random_state = 42;
        cfg.umap_cfg.min_dist = 0.1; % 0.1, 0.001, 0.0001
        cfg.umap_cfg.n_neighbors = 200; % 20 10 5
        cfg.umap_cfg.metric = 'euclidean';
        cfg.umap_cfg.densmap = false;
        
        %cfg.umap_cfg.negative_sample_rate = 20;
        %cfg.umap_cfg.repulsion_strength = 10;
        cfg.umap_cfg.set_op_mix_ratio = 0.25;
        %cfg.umap_cfg.target = targ;
        %cfg.umap_cfg.n_components = 3;
        cfg.umap_cfg.n_epochs = 200;
        %cfg.umap_cfg.init = 'spectral'; %random
    cfg.cluster = 1;
        cfg.cluster_method = 'WATERSHED';

    [Y_train,embedInfo] = embed_pose2(X_feat_train,cfg);
end

%% re-embed
if doRembed
    % load back cfg, to make sure its the same
    embedInfo = load([anadir '/procInfo_train.mat']);

    % test
    tic
    
    cfg = embedInfo.cfg;
    cfg.anadir = anadir;
    cfg.train = 0;
    cfg.group = idat;
    cfg.umap_cfg.knntype = 'faiss'; % mat, faiss
        cfg.umap_cfg.useGPU = true; % mat, faiss
        
    xpath = dir([featdir2 '/*.mat']);
    xpath = cellfun(@(x) sprintf('%s/%s_feat.mat',featdir2,x(1:end-4)),{datasets.name},'un',0);
    [Y_test,embedInfo_test] = embed_pose2(xpath,cfg,embedInfo);
    
    toc
    
end


fprintf('\n TOTAL TIME: %g\n', toc(START))

%% plot embedding results
if plotEmbedding
    figure
    %[nr,nc] = subplot_ratio(max(idat2)*2);
    nr = 3; nc = 2;
    set_bigfig(gcf,[0.35 0.5])

    strs = {'train','test'};

    if strcmp(monk,'wo')
        thisPlot = 2;
    else
        thisPlot = 1:2;
    end

    % plot
    ns=0;
    hax = [];
    for ii=1:numel(thisPlot)
        if thisPlot(ii)==1
            tmpc = load([anadir '/cluster_train.mat']);
            idat2 = idat(idx_train);
        else
            tmpc = load([anadir '/cluster_test.mat']);
            idat2 = idat;
        end
        
        % plot embedding
        subplot(nr,nc,ii)
        hout = plot_embedding(tmpc.outClust,tmpc.Lbnds,tmpc.Ld,1,1);
        title(strs{thisPlot(ii)})
        
        hax(ii,1) = gca;
        
        % plot specificty
        nclust = accumarray(tmpc.clabels,1);
        [n,xe,ye] = histcounts2(tmpc.clabels,idat2,[max(tmpc.clabels) max(idat2)]);
        p = n ./ nclust;

        nclust2 = accumarray(tmpc.clabels,1);
        p2 = nclust2 ./ sum(nclust2(:));

        %p = p ./ accumarray(idat,1)';
        th = 1/max(idat);
        b = mean(abs(p-th),2) * 100;
        %b = sum(p2.*b);
    
        subplot(nr,nc,ii+nc)
        plot(b)
        
        s = sprintf('mean data specificity, weighted by cluster density\n mu=%.3g',mean(b));
        title(s)
        xlabel('cluster ID')
        ylabel('weighted specficity')
        
        hax(ii,2) = gca;

        % plot specificity on embedding map
        tmpc2 = tmpc;
        d = tmpc2.outClust.dens2;
        for jj=1:max(tmpc.clabels)
            sel = tmpc2.Ld==jj;
            d(sel) = b(jj);
        end
        tmpc2.outClust.dens2 = d;
        
        subplot(nr,nc,ii+nc*2)
        hout = plot_embedding(tmpc2.outClust,tmpc2.Lbnds,tmpc2.Ld,1,1);
        title(strs{thisPlot(ii)})
        
        hax(ii,3) = gca;
        foo=1;
    end
    setaxesparameter(hax(:,1),'xlim')
    setaxesparameter(hax(:,1),'ylim')
    setaxesparameter(hax(:,1),'clim')
    
    setaxesparameter(hax(:,3),'xlim')
    setaxesparameter(hax(:,3),'ylim')
    setaxesparameter(hax(:,3),'clim')
    
    % save
    sname = [anadir '/Figures/embedding.pdf'];
    save2pdf(sname,gcf)
end

%% make example videos?
if makeExampleVideos
    dstpath = [anadir '/Vid_clusters'];
    
    % load
    info = load([anadir '/info.mat']);
    clusterInfo = load([anadir '/cluster_train.mat']);
    
    % collapse across monkeys?
    idat_train = info.idat(info.idx_train);
    
    % make videos
    ST = tic;
    parfor id=1:numel(info.datasets)
        try
                tic
                sel = idat_train==id;
                C = clusterInfo.clabels(sel);
                %f = 1:numel(C); % force clips of training segments
                f = frame_train(sel);

                % vid anems
                name = info.datasets(id).name;
                vidnames = {'vid_18261112_full.mp4','vid_18261030_full.mp4'};
                vidnames = cellfun(@(x) [fileparts(anadir) '/' name(1:end-9) '/vids/' x],vidnames,'un',0);


                % call
                vcfg = [];
                vcfg.dThresh = 5;
                vcfg.nrand = 10;
                vcfg.dstpath = dstpath;
                vcfg.suffix = sprintf('id%g',id);
                vcfg.vidnames = vidnames;
                cluster_example_videos(C,f,vcfg)
                toc
                fprintf('\n')
        catch
            error('error on %g',id)
        end
    end
    
    fprintf('TOTAL VIDEO TIME: %g',toc(ST))
end


%% //////////////////////////////////////////////////////////////////////
% ////////////////////          MISC            /////////////////////////
% //////////////////////////////////////////////////////////////////////

function [out,procInfo] = run_proc(name,datadir,featdir,theseFeatures,theseFeatures2,procInfo)

folds = {'data_ground','data_scale'};

% load
in = [];
for ii=1:numel(folds)
    tmp = load([datadir '/' folds{ii} '/' name]);
    in = cat(1,in,tmp);
end

% process
[xf,feat_labels,ifeat,procInfo{1}] = build_pose_features_new(in(1).data_proc,theseFeatures,procInfo{1});
[xf2,feat_labels2,ifeat2,procInfo{2}] = build_pose_features_new(in(2).data_proc,theseFeatures2,procInfo{2});

% combine
X_feat = [xf, xf2];
feat_labels = [feat_labels, feat_labels2];
ifeat = [ifeat, cellfun(@(x) x+ifeat{end}(end),ifeat2,'un',0)];

% output
out = [];
out.X_feat = X_feat;
out.feat_labels = feat_labels;
out.ifeat = ifeat;
out.frame = in(1).data_proc.frame;
out.com = in(1).data_proc.com;
out.info = in(1).data_proc.info;
out.procInfo = procInfo;
out.ifeat = ifeat;
out.data = in;
out.labels = in(1).data_proc.labels;

tmp = rmfield(out,'data');
sname = [featdir '/' name(1:end-4) '_feat.mat'];
parsave(sname,tmp)


