function embed_pose(ecfg)

[parentdir,~,pyenvpath,~,binpath,codepath] = set_pose_paths(0);

% checks
ecfg = checkfield(ecfg,'anadir','needit');
ecfg = checkfield(ecfg,'monk','needit');
ecfg = checkfield(ecfg,'baseDataset','needit');
ecfg = checkfield(ecfg,'nparallel',15);

ecfg = checkfield(ecfg,'train','needit');
ecfg = checkfield(ecfg,'calcFeatures',1);
ecfg = checkfield(ecfg,'normFeatures',2);
ecfg = checkfield(ecfg,'embed',1);
ecfg = checkfield(ecfg,'cluster',1);


% flags
plotEmbedding = 1;
makeExampleVideos = 0;

% settings
nparallel = ecfg.nparallel;
baseDataset = ecfg.baseDataset; %'yo_2021-02-25_01_enviro';

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
featdir_norm = [featdir '_norm'];
if ~exist(featdir_norm); mkdir(featdir_norm); end
    
featInfoPath = [anadir '/featInfo.mat'];
infopath = [anadir '/info.mat'];

% deffine datasets
[datasets,taskInfo] = get_datasets(ecfg.monk);

START = tic;

%% for each dataset, load and build features    
if ecfg.calcFeatures % calculate fresh

    % run the first dataset to get all feature params
    if ecfg.train
        featInfo = {[],[]};
        
        firstDataset = find(contains({datasets.name},baseDataset));
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
        tmp.baseDataset = baseDataset;

        parsave(featInfoPath,tmp)
    else
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
    if ecfg.normFeatures>0
        fprintf('normalizing...\n')
        tic
        if ecfg.normFeatures==1 % independent norm
            if 1
                X_feat = zscore_robust(X_feat);
            elseif 0
                X_feat = zscore(X_feat);
            else
                X_feat = X_feat - nanmean(X_feat);
            end
        elseif ecfg.normFeatures==2 % set norm
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
        fprintf('resaving each one: ')
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
    if ecfg.normFeatures>0
        tmpdir = featdir_norm;
    else
        tmpdir = featdir;
    end
    
    [X_feat,tmpdat] = load_features(tmpdir,'feat',datasets);
    evals(tmpdat); % put into environment
end    

%embed+clust
%{
%% run embedding
if ecfg.embed
    if ecfg.train

        % ----------------------------------------------------
        % select training samples

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
        
        % ----------------------------------------------------
        % run emebdding

        cfg = [];
        cfg.anadir = anadir;
        cfg.min_dist = 0.1; % 0.1, 0.001, 0.0001
        cfg.n_neighbors = 200; % 20 10 5
        cfg.metric = 'euclidean';
        cfg.set_op_mix_ratio = 0.25;
        cfg.n_epochs = 200;
            
        [Y,procInfo] = call_umap(X,cfg);
    end
    
    % now re-embed
    % load back cfg, to make sure its the same
    embedInfo = load([anadir '/procInfo_train.mat']);

    % test
    tic
    
    cfg = embedInfo.cfg;
    cfg.anadir = anadir;
    cfg.train = 0;
    cfg.group = idat;
    cfg.knntype = 'faiss'; % mat, faiss
        cfg.useGPU = true; % mat, faiss
        
    xpath = dir([featdir_norm '/*.mat']);
    xpath = cellfun(@(x) sprintf('%s/%s_feat.mat',featdir_norm,x(1:end-4)),{datasets.name},'un',0);
    [Y_test,embedInfo_test] = embed_pose2(xpath,cfg,embedInfo);
    
    toc
    
end


%% cluster
if cfg.cluster
    if cfg.train
        fprintf('clustering via %s\n',cfg.cluster_method)
        
        tic
        [clabels, Ld, Lbnds,outClust] = examine_clusters(Y, cfg.cluster_method);
        toc
        
        sname = [anadir '/cluster_train.mat'];
        save(sname,'clabels','Ld', 'Lbnds','outClust')
    else
        fprintf('clustering test data using KNN...\n')
        
        tic
        % test data
        cluster_train = load([anadir '/cluster_train.mat']);
        umap_test = load([anadir '/umap_test.mat']);
        umap_train = load([anadir '/umap_train.mat']);
        Y = umap_test.embedding_;
        Y_train = umap_train.embedding_;
        
        % find the state labels
        fprintf('\t finding state labels...\n')
        xv = cluster_train.outClust.xv;
        yv = cluster_train.outClust.yv;

        if 0
            clabels_test = double(interp2(xv,yv,cluster_train.Ld,Y(:,1),Y(:,2),'nearest'));
        else
            IDX = knnsearch(Y_train,Y,'K',10);
            clabels_test = cluster_train.clabels(IDX);
            clabels_test = mode(clabels_test,2);
        end

        % get heatmap again
        fprintf('\t getting density estimate...\n')
        [tmp1,tmp2] = meshgrid(xv,yv);
        [dens, ~, bw] = ksdens(Y,[tmp1(:) tmp2(:)]);
       
        fprintf('\t saving cluster info...\n')
        cluster_test = cluster_train;
        cluster_test.clabels = clabels_test;
        cluster_test.outClust.dens2 = dens;

        sname = [anadir '/cluster_test.mat'];
        save(sname,'-struct','cluster_test')

        toc
    end
end


fprintf('\n TOTAL TIME: %g\n', toc(START))

%% plot embedding results
if plotEmbedding
    figure
    %[nr,nc] = subplot_ratio(max(idat2)*2);
    nr = 3; nc = 2;
    set_bigfig(gcf,[0.35 0.5])

    strs = {'train','test'};

    if ecfg.train
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
%}

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


