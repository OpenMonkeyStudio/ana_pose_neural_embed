%{
- tested on freyr
- master directoery is /mnt/scratch/BV_embed/P_neural_final
- assumes pose data (json files) are in Data_json
- assumes ephys data for each session is in the master folder
%}

%% settings
datadir = '/mnt/scratch/BV_embed/P_neural_final';

monks = {'yo','wo'};

%% run pose preprocessing for both monks
if 0
    procAll = 0; % just proc what we need
    
    logpath = '/mnt/scratch/git/ana_pose_neural_embed/docs/data_log_woodstock_ephys.xlsx';
    res_proc_wo = run_preproc_diffTransform('wo',procAll,logpath);

    logpath = '/mnt/scratch/git/ana_pose_neural_embed/docs/data_log_yoda_ephys.xlsx';
    res_proc_yo = run_preproc_diffTransform('yo',procAll,logpath);
end

%% prepare paths
anadirs = {};
for im=1:numel(monks)
    s = [datadir '/embed_rhesus_' monks{im}];
    if ~exist(s); mkdir(s); end
    anadirs{im} = s;
end

%% extract neural time series

% first, extract all regressors
if 0
BAD_yo = get_all_regressors('yo',datadir);
BAD_wo = get_all_regressors('wo',datadir);
end

% now, get neural time series for each cell
for im=1:numel(monks)
    get_matched_sdf(anadirs{im},monks{im})
end

%% run the embedding

% yoda: train and re-embed

% woodstock: copy params from yoda, and re-embed


%% now analysis

% load

% embedding analysis

% neural analysis

% mebedding+neural analysis