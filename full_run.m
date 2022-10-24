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

runGraphCluster = 1;

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
    BAD_yo = get_all_regressors('yo',datadir);
    BAD_wo = get_all_regressors('wo',datadir);
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

im = 1;
anadir = anadirs{im};

% load
load_pose_neural_data

% summaries

% embedding analysis

% neural analysis

% embedding+neural analysis