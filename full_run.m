%{
- tested on freyr
- master directoery is /mnt/scratch/BV_embed/P_neural_final
- assumes pose data (json files) are in Data_json
- assumes ephys data for each session is in the master folder

- current code is setup to assume everything has been run already and just
need to do analysis/generate figures. to re-run, set appropriate flags
%}

%% add code to path
dbstop if error

%{

% repos
if 0 % 
path1 = 'C:\Users\HaydenLab\Documents\git\oms_internal';
path2 = 'C:\Users\HaydenLab\Documents\git\ana_pose_neural_embed';
ftpath = 'C:\Users\HaydenLab\Documents\_code\fieldtrip-master';
elseif 1 % freyr
path1 = '/mnt/scratch/git/oms_internal';
path2 = '/mnt/scratch/git/ana_pose_neural_embed';
ftpath = '/mnt/scratch/__code/fieldtrip-20210212';
elseif 0 % freyr
path1 = '/mnt/scratch/BV_embed/P_neural_transfer/__code/ana_pose_neural_embed';
path2 = '/mnt/scratch/BV_embed/P_neural_transfer/__code/oms_internal';
ftpath = '/mnt/scratch/__code/fieldtrip-20210212';
end

fprintf('adding repos...\n')
addpath(genpath(path1))
addpath(genpath(path2))

% fieldtrip
fprintf('adding fieldtrip...\n')
rmpath(genpath(ftpath)); % it would have been added
addpath(ftpath)
ft_defaults
global ft_default
ft_detault.trackcallinfo = 'no';
ft_detault.showcallinfo = 'no';

% NB: change paths in set_pose_paths
%}

[parentdir,jsondir,pyenvpath,rpath,binpath,codepath,ephyspath] = set_pose_paths(0);

%% settings
datadir = '/mnt/scratch/BV_embed/P_neural_final_oldEmbed';
%datadir = '/mnt/scratch/BV_embed/P_neural_transfer';
%datadir = 'D:\P_neural_final_oldEmbed';

monks = {'yo','wo'};

% flags
runPosePreproc = 0;
runRegressors = 0;
runSdfMatching = 0;

runEmbedding_firstMonk = 0;
runEmbedding_secondMonk = 0;

runGraphCluster = 0;

runAnalyses = 1;
    anaMonks = [1 2];
    modExamples = [8 3];
    
    makeExampleActionVideos = 0;
    makeExampleModuleVideos = 0;
    anaEmbedding = 0;
    anaBehavHier = 1;
    anaEncoding = 0;
    anaSwitch = 0;

% prepare paths
anadirs = {};
for im=1:numel(monks)
    s = [datadir '/embed_rhesus_' monks{im}];
    if ~exist(s); mkdir(s); end
    anadirs{im} = s;
    
    f = [s '/Figures2'];
    if ~exist(f); mkdir(f); end    
end

%% run pose preprocessing for both monks
if runPosePreproc
    procAll = 0; % just proc what we need
    
    logpath = [ephypath '/data_log_woodstock_ephys.xlsx'];
    res_proc_wo = run_preproc_diffTransform('wo',procAll,logpath);

    logpath = [ephyspath '/data_log_yoda_ephys.xlsx'];
    res_proc_yo = run_preproc_diffTransform('yo',procAll,logpath);
end


%% extract neural time series

% first, extract all regressors
if runRegressors
    BAD_reg = {};
    for im=1:numel(monks)
    	BAD_reg{im} = get_all_regressors(monks{im},datadir);
    end
end

% now, get neural time series for each cell
if runSdfMatching
    for im=1:numel(monks)
        get_matched_sdf(anadirs{im},monks{im})
    end
end

%% run the embedding

% ---------------------------------------------
% yoda: train and re-embed
% ---------------------------------------------
if runEmbedding_firstMonk
    ecfg = [];
    ecfg.anadir = anadirs{1};
    ecfg.monk = 'yo';
    ecfg.nparallel = 15;

    ecfg.calc_features = 0;
        ecfg.base_dataset = 'yo_2021-02-25_01_enviro';
        ecfg.normtype = 2;
    ecfg.get_training_data = 1;
    ecfg.embedding_train = 0;
    ecfg.embedding_test = 1;
        ecfg.knntype = 'faiss';
        ecfg.K = 20;
        ecfg.gputype = 1;
    ecfg.cluster_train = 1;
        ecfg.cluster_method = 'WATERSHED';
    ecfg.cluster_test = 1;

    embed_pose(ecfg);
end


% ---------------------------------------------
% woodstock
% ---------------------------------------------

if runEmbedding_secondMonk
    % copy params from yoda
    fprintf('copying training info from %s to %s... \n',monks{1},monks{2})
    
    theseFiles = {'featInfo.mat',
        'umap_train.mat',
        'umap_cfg.mat',
        'cluster_train.mat'
        };
    
    for ii=1:numel(theseFiles)
        f = theseFiles{ii};
        src = [anadirs{1} '/' f];
        dst = [anadirs{2} '/' f];
        copyfile(src,dst)
    end

    % now embed
    ecfg = [];
    ecfg.anadir = anadirs{2};
    ecfg.monk = 'wo';
    ecfg.nparallel = 15;

    ecfg.calc_features = 0;
        ecfg.base_dataset = 'yo_2021-02-25_01_enviro';
        ecfg.normtype = 2;
    ecfg.get_training_data = 2;
        ecfg.trainingdir = [anadirs{1} '/X_feat_norm'];
    ecfg.embedding_train = 0;
    ecfg.embedding_test = 1;
        ecfg.knntype = 'faiss';
        ecfg.K = 20;
        ecfg.gputype = 1; %0=cpu,1=first gpu, 2=all gpu
    ecfg.cluster_train = 0;
        ecfg.cluster_method = 'WATERSHED';
    ecfg.cluster_test = 1;

    embed_pose(ecfg);
    
end


%% graph clusterin
if runGraphCluster
    ST = tic;
    
    % loop over monks
    for im=1:numel(monks)
        anadir = anadirs{im};
        load_pose_neural_data
        
        tic
        tmp = get_graph_cluster(C,idat,anadir);
        toc
        
        % get peaks, to collapse actions
        if 0
            tic

            nlabel = max(C_train);
            Ld = cluster_train.Ld;
            xv = cluster_train.outClust.xv;
            yv = cluster_train.outClust.yv;

            peaks = [];
            for ic=1:nlabel

                [icol,irow] = find(Ld==ic);

                icol = floor(median(icol));
                irow = floor(median(irow));
                xy = double([xv(irow) yv(icol)]);
                peaks(ic,:) = xy;
            end
            zpeaks = linkage(peaks);

            ncollapse = size(peaks,1)-1:-1:4;

            % loop over each peak
            for ii=1:numel(ncollapse)
                cp = cluster(zpeaks,'maxclust',ncollapse(ii));

                Cp = changem(C,cp,1:nlabel);

                % run graph clustering
                mpathname = sprintf('modularity_test_collapse/naction%g',ncollapse(ii));
                tmp2 = get_graph_cluster(Cp,idat,anadir,1,mpathname);
            end
            toc
        end
        foo=1;
    end
    
    FN = toc(ST);
    fprintf('TOTAL ELAPSED TIME, GRAPHY CLUSTER: %g\n',FN)
end


%% now analysis
if runAnalyses

    SEG_all = {};
    for im = anaMonks
        %im = anaMonks(im1);
        
        % load
        monk = monks{im};
        anadir = anadirs{im};
        figdir = [anadir '/Figures'];
        if ~exist(figdir); mkdir(figdir); end
        
        load_pose_neural_data
        
        % summaries
        ana_summary

        % make videos of all actions
        %
        if makeExampleActionVideos && im==1
            dstpath = [anadir '/Vid_clusters'];

            % make videos
            ST = tic;
            for id=1:numel(datasets)
                try
                    tic
                    sel = idat==id;
                    tmpc = cluster_test.clabels(sel);
                    f = frame(sel);

                    % vid anems
                    name = datasets(id).name;
                    vidnames = {'vid_18261112_full.mp4','vid_18261030_full.mp4'};
                    vidnames = cellfun(@(x) [fileparts(anadir) '/' name '/vids/' x],vidnames,'un',0);

                    % call
                    vcfg = [];
                    vcfg.dThresh = [0.5 10]; %sec
                    vcfg.nrand = 10;
                    vcfg.dstpath = dstpath;
                    vcfg.suffix = sprintf('id%g',id);
                    vcfg.vidnames = vidnames;
                    cluster_example_videos(tmpc,f,vcfg)
                    toc
                    fprintf('\n')
                catch
                    error('error on %g',id)
                end
            end

            fprintf('TOTAL VIDEO TIME: %g',toc(ST))
        end
        
        
        % example videos of modules
        if makeExampleModuleVideos
            dstpath = [anadir '/Vid_modules'];

            % make videos
            id = modExamples(im);
            
            ST = tic;
            try
                tic
                sel = idat==id;
                tmpc = cluster_test.clabels(sel);
                f = frame(sel);

                % re-map actiions to modules
                q = squeeze(res_mod.obs.modularity(1,id,:));
                [~,imx] = max(q);
                newval = res_mod.obs.labels_cut(1,id,imx,:);
                tmpc2 = changem(tmpc,newval,1:nstate);

                % only use segments that have >1 action
                d = find(diff(tmpc2)~=0);
                st = [1; d+1];
                fn = [d; numel(tmpc2)];

                bad = false(numel(st),1);
                for is=1:numel(st)
                    tmp = tmpc(st(is):fn(is));
                    n = numel(unique(tmp));
                    bad(is) = n < 2;
                end

                good = fill_logical(numel(f),[st(~bad), fn(~bad)]);
                f(~good) = [];
                tmpc2(~good) = [];
                tmpc(~good) = [];

                % vid anems
                name = datasets(id).name;
                vidnames = {'vid_18261112_full.mp4','vid_18261030_full.mp4'};
                vidnames = cellfun(@(x) [fileparts(anadir) '/' name '/vids/' x],vidnames,'un',0);

                % call
                vcfg = [];
                vcfg.dThresh = [0.5 30]; %sec
                vcfg.nrand = 15;
                vcfg.dstpath = dstpath;
                vcfg.suffix = sprintf('id%g',id);
                vcfg.vidnames = vidnames;
                cluster_example_videos(tmpc2,f,vcfg)
                toc
                fprintf('\n')
            catch
                error('error on %g',id)
            end

            fprintf('TOTAL VIDEO TIME: %g',toc(ST))
         end
        
         
        %}

        
        % ------------------------------------------------------------
        % embedding analysis
        if anaEmbedding
            % embedding maps (Figure 2B,C)
            % - note: the training embedding will be the same for both
            figure('name',[monks{im} ' embedding'])
            nr = 1; nc = 2;
            set_bigfig(gcf,[0.7 0.4])

            subplot(nr,nc,1)
            hout1 = plot_embedding(cluster_train.outClust,cluster_train.Lbnds,cluster_train.Ld,1,1);
            title([monks{im} ': train'])
            subplot(nr,nc,2)
            hout2 = plot_embedding(cluster_test.outClust,cluster_test.Lbnds,cluster_test.Ld,1,1);
            title([monks{im} ': test'])
            setaxesparameter([hout1.hax, hout2.hax],'clim')
            
            sname = [figdir '/embedding_train_test.pdf'];
            save2pdf(sname,gcf)

            % validity of clusters
            ana_cluster_validity

            % example actions and their embedding (Figure 2D-F)
            if im==1
                plot_embedding_yo_02_25_2021(anadir)
            end
        end
        
        
        % ------------------------------------------------------------
        % behavioural analysis
        if anaBehavHier
            cfg = [];
            cfg.anadir = anadir;
            cfg.plot_modularity_mean = 1;
            cfg.plot_modularity_hist = 1;
            cfg.plot_hierarchy_hist = 1;
            if im==1
                cfg.example_modularity_id = 8; % yo_2021-02-25_01
                cfg.example_hierarchy_id = 8; % yo_2021-02-25_01
            else
                cfg.example_modularity_id = 3; % wo_2021-12-08_01
                cfg.example_hierarchy_id = 3; % wo_2021-12-08_01
            end

            ana_mod_hierarchy(cfg,res_mod) %(Figure 3)
            
            ana_mod_hierarchy_collapse
        end

        % ------------------------------------------------------------
        % action encoding
        if anaEncoding
            % example action encoding (Figure 4A,B)
            if im==1
                name = 'yo_2021-06-08_01_enviro'; % lots of neurons
                plot_example_encoding(datasets,name,figdir,SDF,C,idat)
                
                % makes and save example video
                % viz_example
            end

            % action encoding (Figure 4C,D)
            cfg = [];
            cfg.sdfpath = sdfpath;
            cfg.figdir = figdir;
            cfg.savesuffix = '_test2';
            cfg.datasets = datasets;
            cfg.nstate = nstate;
            cfg.fs_frame = fs_frame;
            cfg.uarea = uarea;
            cfg.get_encoding = 1;
                cfg.testtype = 'kw';
                cfg.nrand = 20;
                cfg.nboot = 1;
                cfg.eng_lim = [3, ceil(1*cfg.fs_frame)];
                cfg.ilag = 1;
                cfg.theseCuts = [2:8 10:2:20 23:3:31, nstate];

            out_encode = ana_action_encoding(cfg,SDF,res_mod,C,iarea,idat);
            
            % predict encoding from elec position (Figure 5)
            cfg = [];
            cfg.monk = monks{im};
            cfg.figdir = figdir;
            cfg.model_binned = 0;
            cfg.varname = 'kwF';
            cfg.uarea = uarea;

            Var = out_encode.A_act(:,:,1);
            out_depth_encode = ana_depth_corr(cfg,Var,SDF,iarea);

            nbins = [5 10 15 20];
            for ib=1:numel(nbins)
                cfg.model_binned = 1;
                cfg.nbin = nbins(ib);
                ana_depth_corr(cfg,Var,SDF,iarea);
            end
            
            % --------------------------------------------------------
            % control: action encoding, unique cells
            [isFirst,area,days] = select_unique_cells(monk);
            [ich,iday] = find(isFirst);

            ch = [SDF.ch];
            day = {SDF.day};
            good = false(size(SDF));
            for ii=1:numel(ich)
                sel = ch==ich(ii) & strcmp(day,days{iday(ii)});
                good(sel) = 1;
            end

            % cull
            SDF2 = SDF(good);
            sdfnames2 = sdfnames(good);
            iarea2 = iarea(good);
            areas2 = areas(good);
            
            % action encoding
            figdir2 = [figdir '_onlyFirstCell'];
            if ~exist(figdir2); mkdir(figdir2); end
            
            cfg = [];
            cfg.sdfpath = sdfpath;
            cfg.figdir = figdir2;
            cfg.savesuffix = '_onlyFirst';
            cfg.datasets = datasets;
            cfg.nstate = nstate;
            cfg.fs_frame = fs_frame;
            cfg.uarea = uarea;
            cfg.get_encoding = 1;
                cfg.testtype = 'kw';
                cfg.nrand = 20;
                cfg.nboot = 1;
                cfg.eng_lim = [3, ceil(1*cfg.fs_frame)];
                cfg.ilag = 1;
                cfg.theseCuts = [2:8 10:2:20 23:3:31, nstate];

            out_encode_first = ana_action_encoding(cfg,SDF2,res_mod,C,iarea2,idat);
            
             % predict encoding from elec position
            cfg = [];
            cfg.monk = monks{im};
            cfg.figdir = figdir2;
            cfg.model_binned = 0;
            cfg.varname = 'kwF';
            cfg.uarea = uarea;

            Var = out_encode_first.A_act(:,:,1);
            out_depth_encode_first = ana_depth_corr(cfg,Var,SDF2,iarea2);

            
        end
        
        % ------------------------------------------------------------
        % action switch encoding
        if anaSwitch
            % switch sdf (Figure 6B)
            cfg = [];
            cfg.sdfpath = sdfpath;
            cfg.figdir = [figdir '_controlSwitch'];
            cfg.datasets = datasets;
            cfg.fs_frame = fs_frame;
            cfg.uarea = uarea;

            cfg.only_nonengage = 0;
            cfg.plot_lim = [-1 1];
            cfg.seg_lim = [-1 1];
            cfg.seg_min = 0.2; % 0.2
            cfg.eng_smooth = 1;

            cfg.avgtype = 'median';
            cfg.normtype = 'preswitch';
            cfg.weighted_mean = 0;

            out_switch = ana_switch_sdf(cfg,SDF,C,frame,iarea,idat);

            % switch sdf + controls (Figure 6C)
            cfg.only_nonengage = 1;
            cfg.weighted_mean = 1;
            out_switch2 = ana_switch_sdf(cfg,SDF,C,frame,iarea,idat);

            % predict switching signal from elec location
            cfg = [];
            cfg.monk = monks{im};
            cfg.figdir = figdir;
            cfg.model_binned = 0;
            cfg.varname = 'dSwitch';
            cfg.uarea = uarea;

            Var = out_switch.dSeg;
            out_depth_switch = ana_depth_corr(cfg,Var,SDF,iarea);

            % store for later
            SEG_all{im,1} = out_switch;
            SEG_all{im,2} = out_switch2;
        end
    end

    % ------------------------------------------------------------
    % plot segment, collapsed for yoda and wood (Figure 6A)
    if anaSwitch
        figdir2 = [anadirs{1} '/Figures_bothMonk'];
        if ~exist(figdir2); mkdir(figdir2); end

        % plot
        iseg = 1;
        res_tmp = [SEG_all{1,iseg}.RES_seg; SEG_all{2,iseg}.RES_seg];

        cfg = [];
        cfg.figdir = figdir2;
        cfg.fs_frame = fs_frame;
        cfg.uarea = uarea;

        cfg.plot_lim = SEG_all{1,iseg}.cfg.plot_lim;
        cfg.seg_lim = SEG_all{1,iseg}.cfg.seg_lim;
        cfg.avgtype = 'median';
        cfg.normtype = 'preswitch';
        cfg.weighted_mean = 0;

        tmp = ana_switch_sdf(cfg,res_tmp);
    end    
end
