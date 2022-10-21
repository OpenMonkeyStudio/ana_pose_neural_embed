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
runEmbeddingTrain = 1;


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

    ecfg.train = 1;
    ecfg.baseDataset = 'yo_2021-02-25_01_enviro';
    ecfg.calcFeatures = 0;
    ecfg.normFeatures = 2;
    ecfg.embed = 1;
    ecfg.cluster = 1;

    embed_pose(ecfg);
end

% ---------------------------------------------
% woodstock
% ---------------------------------------------

% copy params from yoda

% re-embed

% re-cluster

%% now analysis

% load

% embedding analysis

% neural analysis

% mebedding+neural analysis