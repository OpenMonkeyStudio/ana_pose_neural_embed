function embed_pose(ecfg)
% embed_pose(ecfg)

[parentdir,~,pyenvpath,~,binpath,codepath] = set_pose_paths(0);

% checks
ecfg = checkfield(ecfg,'anadir','needit');
ecfg = checkfield(ecfg,'monk','needit');
ecfg = checkfield(ecfg,'base_dataset','needit');
ecfg = checkfield(ecfg,'nparallel',15);

ecfg = checkfield(ecfg,'calc_features',1);
    ecfg = checkfield(ecfg,'normtype',2);
    ecfg = checkfield(ecfg,'base_dataset',2);
ecfg = checkfield(ecfg,'embedding_train',1);
ecfg = checkfield(ecfg,'embedding_test',1);
    ecfg = checkfield(ecfg,'knntype','faiss');
    ecfg = checkfield(ecfg,'K',20);
ecfg = checkfield(ecfg,'cluster_train',1);
ecfg = checkfield(ecfg,'cluster_test',1);

ecfg = checkfield(ecfg,'plot_embedding',1);


% paths
anadir = ecfg.anadir;
datadir = [fileparts(anadir) '/Data_proc_13joint'];

featdir = [anadir '/X_feat'];
if ~exist(featdir); mkdir(featdir); end
featdir_norm = [featdir '_norm'];
if ~exist(featdir_norm); mkdir(featdir_norm); end
    
featInfoPath = [anadir '/featInfo.mat'];
infopath = [anadir '/info.mat'];

% deffine datasets
[datasets,taskInfo] = get_datasets(ecfg.monk);


% settings
nparallel = ecfg.nparallel;
base_dataset = ecfg.base_dataset; %'yo_2021-02-25_01_enviro';

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

% umap settings
ucfg = [];
ucfg.anadir = anadir;
ucfg.min_dist = 0.1; % 0.1, 0.001, 0.0001
ucfg.n_neighbors = 200; % 20 10 5
ucfg.metric = 'euclidean';
ucfg.set_op_mix_ratio = 0.75;
ucfg.n_epochs = 200;


START = tic;

%% for each dataset, load and build features    
if ecfg.calc_features>0 % calculate fresh
    if ecfg.calc_features==1 % first subject, first dataset
        % run the first dataset to get all feature params
        featInfo = {[],[]};
        
        firstDataset = find(contains({datasets.name},base_dataset));
        if numel(firstDataset)~=1; error('unrecognized dataset'); end

        name = datasets(firstDataset).name;
        [out,featInfo] = run_proc(name,datadir,featdir,theseFeatures,theseFeatures2,featInfo);

        % save it
        tmp = [];
        tmp.featInfo = featInfo;
        tmp.datasets = datasets;
        tmp.theseFeatures = theseFeatures;
        tmp.theseFeatures2 = theseFeatures2;
        tmp.feat_labels = out.feat_labels;
        tmp.ifeat = out.ifeat;
        tmp.base_dataset = base_dataset;

        parsave(featInfoPath,tmp)
    else % other subjects
        tmp = load(featInfoPath);
        featInfo = tmp.procInfo;
        firstDataset = [];

        % resave info for this dataset
        tmp.datasets = datasets;
        parsave(featInfoPath,tmp)
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

        run_proc(name,datadir,featdir,theseFeatures,theseFeatures2,featInfo);
    end
    toc
    
    % load and concatenate them
    [X_feat,tmpdat] = load_features(featdir,'feat',datasets);
    evals(tmpdat); % put into environment
    
    % save the processing info
    save(infopath,'-struct','tmpdat')
    
    % normalize them
    if ecfg.normtype>0
        fprintf('normalizing...\n')
        tic
        if ecfg.normtype==1 % independent norm
            if 1
                X_feat = zscore_robust(X_feat);
            elseif 0
                X_feat = zscore(X_feat);
            else
                X_feat = X_feat - nanmean(X_feat);
            end
        elseif ecfg.normtype==2 % set norm
            for jj=1:numel(ifeat)
                tmp = X_feat(:,ifeat{jj});
                z = zscore_robust(tmp,[],'all');
                X_feat(:,ifeat{jj}) = z;
            end
        else
            error('huh?')
        end
        toc

        % resave
        tic
        fprintf('resaving each normalized dataset: ')
        for id=1:numel(datasets)
            name = datasets(id).name;
            %fprintf('%g: %s\n',id,name)
            fprintf('%g,',id)

            name_in = [featdir '/' name '_feat.mat'];
            tmp_in = load(name_in);    

            % resave
            x = X_feat(idat==id,:);

            tmp_out = tmp_in;
            tmp_out.X_feat = x;

            name_out = [featdir_norm '/' name '_feat.mat'];
            parsave(name_out,tmp_out)
        end
        fprintf('\n')
        toc    
    end
else 
    fprintf('reloading features... \n')
    
    % load data info
    load(infopath)
    
    % load features
    if ecfg.normtype>0; tmpdir = featdir_norm;
    else; tmpdir = featdir;
    end
    
    [X_feat,tmpdat] = load_features(tmpdir,'feat',datasets);
    evals(tmpdat); % put into environment
end    

%% run embedding
if ecfg.embedding_train
    % ----------------------------------------------------
    % select training samples
    fprintf('constructing training set... \n')
    tic
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
    %idat_train = idat(idx_train);
    %frame_train = frame(idx_train);

    % save
    save(infopath,'-append','idx_train')
    toc

    % ----------------------------------------------------
    % run emebdding training
    tic

    [Y_train,ucfg] = call_umap(X_feat_train,ucfg);

    % save cfg for easy loading
    sname = [anadir '/umap_cfg.mat'];
    save(sname,'-struct','ucfg')

    toc
else
    fprintf('reloading trained embedding... \n')
    
    sname = [anadir '/umap_cfg.mat'];
    ucfg = load(sname);

    tmp = load([anadir '/umap_train.mat']);
    Y_train = tmp.embedding_;
    
    X_feat_train = X_feat(idx_train,:);

end
    

%% re-embed
if ecfg.embedding_test
    tic
    
    % call faiss
    if strcmp(ecfg.knntype,'faiss')
        Xq = cellfun(@(x) sprintf('%s/%s_feat.mat',featdir_norm,x),{datasets.name},'un',0);

            
        fprintf('calling faiss KNN...\n')
        dat = [];
        dat.X = X_feat_train;
        dat.Xq = Xq;
        dat.K = ecfg.K;
        dat.metric = ucfg.metric;
        dat.useGPU = 1; % 1=single core, 2=all cores

        opts = [];
        opts.func = [codepath '/python/call_knnsearch.py'];
        opts.verbose = 0;
        opts.clean_files = 1;
        opts.pyenvpath = pyenvpath;

        out = sendToPython(dat,opts);

        % final output (make sure its in the proper format)
        if ~iscell(out.distances)
            sz = size(out.neighbors);

            IDX = reshape( permute(out.neighbors,[2 1 3]), [sz(1)*sz(2) sz(3)] ) + 1;
            D = reshape( permute(out.distances,[2 1 3]), [sz(1)*sz(2) sz(3)] );
        else
            IDX = cat(1,out.neighbors{:})+1;
            D = cat(1,out.distances{:});
        end
    else % default to matlab
        fprintf('\t matlab single core ...\n')
        [IDX,D] = knnsearch(X_feat_train,X_feat,'K',ecfg.K,'distance',ucfg.metric);
    end

    % get the embeding points
    tmp_test = Y_train(IDX,:);
    tmp_test = reshape(tmp_test,[numel(idat), ecfg.K, size(Y_train,2)]);
    Y_test = permute(nanmedian(tmp_test,2),[1 3 2]);

    % save
    umap_test = [];
    umap_test.embedding_ = Y_test;
    umap_test.K = ecfg.K;
    umap_test.type = 'knnsearch';
    umap_test.metric = ucfg.metric;
    umap_test.knn_idx = IDX;
    umap_test.D_train = D;
    sname = [anadir '/umap_test.mat'];
    save(sname,'-struct','umap_test','-v7.3')

    toc
end

foo=1;

%% cluster
if ecfg.cluster_train
    fprintf('clustering via %s\n',ecfg.cluster_method)

    tic
    [clabels, Ld, Lbnds,outClust] = examine_clusters(Y_train, ecfg.cluster_method);
    toc

    cluster_train = [];
    cluster_train.clabels = clabels;
    cluster_train.Lbnds = Lbnds;
    cluster_train.Ld = Ld;
    cluster_train.outClust = outClust;
    
    sname = [anadir '/cluster_train.mat'];
    save(sname,'-struct','cluster_train')
end

%% assign cluster based on trained data
if ecfg.cluster_test
    fprintf('clustering test data...\n')

    tic
    % test data
    fprintf('\t reloading: \n')
    if ~exist('cluster_train')==1; fprint('cluster,'); cluster_train = load([anadir '/cluster_train.mat']); end
    if ~exist('umap_test')==1; fprint('umap_test,'); umap_test = load([anadir '/umap_test.mat']); end
    Y_test = umap_test.embedding_;

    % find the state labels
    fprintf('\t finding state labels...\n')
    clabels_test = cluster_train.clabels(umap_test.knn_idx);
    clabels_test = mode(clabels_test,2);

    % get heatmap again
    fprintf('\t getting density estimate...\n')
    xv = cluster_train.outClust.xv;
    yv = cluster_train.outClust.yv;
    [tmp1,tmp2] = meshgrid(xv,yv);
    [dens, ~, bw] = ksdens(Y_test,[tmp1(:) tmp2(:)]);

    fprintf('\t saving cluster info...\n')
    cluster_test = cluster_train;
    cluster_test.clabels = clabels_test;
    cluster_test.outClust.dens2 = dens;

    sname = [anadir '/cluster_test.mat'];
    save(sname,'-struct','cluster_test')

    toc
end

%% finish and clean
delete(gcp('nocreate'))
fprintf('\n TOTAL TIME: %g\n', toc(START))

%% plot embedding results
if ecfg.plot_embedding
    fprintf('\n plotting...\n')
    figure
    %[nr,nc] = subplot_ratio(max(idat2)*2);
    nr = 3; nc = 2;
    set_bigfig(gcf,[0.35 0.5])

    strs = {'train','test'};

    if ecfg.calc_features<2
        thisPlot = 1:2;
    else
        thisPlot = 2;
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
    figdir = [anadir '/Figures'];
    if ~exist(figdir); mkdir(figdir); end
    sname = [figdir '/embedding.pdf'];
    save2pdf(sname,gcf)
end


%% //////////////////////////////////////////////////////////////////////
% ////////////////////          MISC            /////////////////////////
% //////////////////////////////////////////////////////////////////////

function [out,featInfo] = run_proc(name,datadir,featdir,theseFeatures,theseFeatures2,featInfo)

folds = {'data_ground','data_scale'};

% load
in = [];
for ii=1:numel(folds)
    tmp = load([datadir '/' folds{ii} '/' name '_proc.mat']);
    in = cat(1,in,tmp);
end

% process
[xf,feat_labels,ifeat,featInfo{1}] = build_pose_features_new(in(1).data_proc,theseFeatures,featInfo{1});
[xf2,feat_labels2,ifeat2,featInfo{2}] = build_pose_features_new(in(2).data_proc,theseFeatures2,featInfo{2});

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
out.featInfo = featInfo;
out.ifeat = ifeat;
out.data = in;
out.labels = in(1).data_proc.labels;

tmp = rmfield(out,'data');
sname = [featdir '/' name '_feat.mat'];
parsave(sname,tmp)


