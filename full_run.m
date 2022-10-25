%{
- tested on freyr
- master directoery is /mnt/scratch/BV_embed/P_neural_final
- assumes pose data (json files) are in Data_json
- assumes ephys data for each session is in the master folder
%}

%% settings
datadir = '/mnt/scratch/BV_embed/P_neural_final';

monks = {'yo','wo'};

% flags
runPosePreproc = 0;
runRegressors = 0;
runSdfMatching = 0;
runEmbeddingTrain = 0;
runEmbeddingTest = 0;

runGraphCluster = 0;

% prepare paths
anadirs = {};
for im=1:numel(monks)
    s = [datadir '/embed_rhesus_' monks{im}];
    if ~exist(s); mkdir(s); end
    anadirs{im} = s;
end

%% run pose preprocessing for both monks
if runPosePreproc
    procAll = 0; % just proc what we need
    
    logpath = '/mnt/scratch/git/ana_pose_neural_embed/docs/data_log_woodstock_ephys.xlsx';
    res_proc_wo = run_preproc_diffTransform('wo',procAll,logpath);

    logpath = '/mnt/scratch/git/ana_pose_neural_embed/docs/data_log_yoda_ephys.xlsx';
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
if runEmbeddingTrain
    ecfg = [];
    ecfg.anadir = anadirs{1};
    ecfg.monk = 'yo';
    ecfg.nparallel = 15;

    ecfg.calc_features = 0;
        ecfg.base_dataset = 'yo_2021-02-25_01_enviro';
        ecfg.normtype = 2;
    ecfg.embedding_train = 0;
    ecfg.embedding_test = 1;
        ecfg.knntype = 'faiss';
        ecfg.K = 20;
    ecfg.cluster_train = 1;
        ecfg.cluster_method = 'WATERSHED';
    ecfg.cluster_test = 1;

    embed_pose(ecfg);
end


% ---------------------------------------------
% woodstock
% ---------------------------------------------

if runEmbeddingTest
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
        copyfiles(src,dst)
    end

    % now embed
    ecfg = [];
    ecfg.anadir = anadirs{2};
    ecfg.monk = 'wo';
    ecfg.nparallel = 15;

    ecfg.calc_features = 1;
        ecfg.base_dataset = 'yo_2021-02-25_01_enviro';
        ecfg.normtype = 2;
    ecfg.embedding_train = 0;
    ecfg.embedding_test = 1;
        ecfg.knntype = 'faiss';
        ecfg.K = 20;
    ecfg.cluster_train = 0;
        ecfg.cluster_method = 'WATERSHED';
    ecfg.cluster_test = 1;

    embed_pose(ecfg);
    
end


%% graph clusterin
if runGraphCluster
    for im=1%:numel(monks)
        anadir = anadirs{im};
        load_pose_neural_data
        tmp = get_graph_cluster(C,idat,anadir);
    end
end


%% now analysis
DAT_all = {};

for im=1:numel(monks)
    anadir = anadirs{im};

    % load
    load_pose_neural_data

    % summaries

    % ------------------------------------------------------------
    % embedding analysis
    ana_mod_hierarchy


    % ------------------------------------------------------------
    % embedding+neural analysis

    % action encoding
    cfg = [];
    cfg.sdfpath = sdfpath;
    cfg.figdir = figdir;
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

    % predict encoding from elec position
    cfg = [];
    cfg.monk = monks{im};
    cfg.figdir = figdir;
    cfg.model_grid = 0;
    cfg.varname = 'kwF';
    cfg.uarea = uarea;

    Var = out_encode.A_act(:,:,1);
    out_depth_encode = ana_depth_corr(cfg,Var,SDF,iarea);

    % switch sdf
    cfg = [];
    cfg.sdfpath = sdfpath;
    cfg.figdir = figdir;
    cfg.datasets = datasets;
    cfg.fs_frame = fs_frame;
    cfg.uarea = uarea;

    cfg.only_nonengage = 0;
    cfg.seg_lim = [-1 1];
    cfg.seg_min = 0.2;
    cfg.eng_smooth = 1;

    cfg.avgtype = 'median';
    cfg.normtype = 'presegnorm';
    cfg.weighted_mean = 0;

    out_switch = ana_switch_sdf(cfg,SDF,C,frame,iarea,idat);

    % switch sdf + controls 
    cfg.only_nonengage = 1;
    cfg.weighted_mean = 1;
    out_switch2 = ana_switch_sdf(cfg,SDF,C,frame,iarea,idat);

    % predict switching signal from elec location
    cfg = [];
    cfg.monk = monks{im};
    cfg.figdir = figdir;
    cfg.model_grid = 0;
    cfg.varname = 'dSwitch';
    cfg.uarea = uarea;

    Var = out_switch.dSeg;
    out_depth_switch = ana_depth_corr(cfg,Var,SDF,iarea);

    % store for later
    DAT_all{im,1} = out_switch;
end

% plot segment, collapsed for yoda and wood
figdir2 = [anadirs{1} '/Figures_bothMonk'];
if ~exist(figdir2); mkdir(figdir2); end

cfg = [];
cfg.figdir = figdir2;
cfg.fs_frame = fs_frame;
cfg.uarea = uarea;

cfg.avgtype = 'median';
cfg.normtype = 'presegnorm';
cfg.weighted_mean = 0;

res_tmp = [DAT_all{1,1}.RES_seg; DAT_all{2,1}.RES_seg];
tmp = ana_switch_sdf(cfg,res_tmp);
