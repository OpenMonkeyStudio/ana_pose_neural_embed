%% settings
datadir = '/mnt/scratch/BV_embed/P_neural_final';

%% run pose preprocessing for both monks
if 1
    procAll = 0; % just proc what we need
    
    logpath = '/mnt/scratch/git/ana_pose_neural_embed/docs/data_log_woodstock_ephys.xlsx';
    res_proc_wo = run_preproc_diffTransform('wo',procAll,logpath);

    logpath = '/mnt/scratch/git/ana_pose_neural_embed/docs/data_log_yoda_ephys.xlsx';
    res_proc_yo = run_preproc_diffTransform('yo',procAll,logpath);
end

%% extract neural time series
BAD_yo = get_all_regressors('yo',datadir);
BAD_yo = get_all_regressors('wo',datadir);

