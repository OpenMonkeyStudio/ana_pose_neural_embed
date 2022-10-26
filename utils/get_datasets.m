function [datasets,taskInfo] = get_datasets(monk)
% [datasets,taskInfo] = get_datasets(monk)

% get datasets
if strcmp(monk,'yo')
    %s = [get_code_path() '/bhv_cluster/data_log_yoda_ephys_bv.xlsx'];
    s = '/mnt/scratch/git/ana_pose_neural_embed/docs/data_log_yoda_ephys.xlsx';
elseif strcmp(monk,'wo')
    s = '/mnt/scratch/git/ana_pose_neural_embed/docs/data_log_woodstock_ephys.xlsx';
else
    error('unrecognized monkey')
end

taskInfo = readtable(s);
datasets = struct('name',taskInfo.name);

%datasets = datasets(8:18); % testing