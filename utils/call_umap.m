function [Y,procInfo] = call_umap(X,cfg)

% checks
cfg = checkfield(cfg,'anadir','needit');

% extract
anadir = cfg.anadir;

% define paths
[~,~,pyenvpath] = set_pose_paths(0);


% save input
% procInfo.grp = grp;
% procInfo.igrp = igrp;
% procInfo.cfg = cfg;
% save(sname_procInfo,'-struct','procInfo')


%% call embeddiing
verbose = 1;

tic
disp('embedding train...')

% prepare data
umap_cfg = rmfield(cfg,'anadir');
umap_cfg.X_train = X;
umap_cfg.func = 'fit';

% supervised?
if ( isfield(umap_cfg,'target') && all(umap_cfg.target==-1) ) || (isfield(umap_cfg,'target') && ~isempty(umap_cfg.target) )
    umap_cfg = rmfield(umap_cfg,'target');      
end

% python inputs
name_in = [anadir '/umap_data_train.mat'];
savepath = anadir;

% send to python
fprintf('\t presaving for transfer \n')
save(name_in,'-struct','umap_cfg','-v7.3')

func = [get_code_path() '/bhv_cluster/matlab/python/call_umap.py'];

commandStr = sprintf('%s %s %s %s',pyenvpath,func,name_in,savepath);
fprintf('\t executing: %s \n',commandStr)

if verbose
    [status,result] = system(commandStr,'-echo');
else
    [status,result] = system(commandStr);
end

% wass it succesful?
if status~=0 % fail
    error('FAIL')
end
toc

% load back
umap_train = load([savepath '/umap_train.mat']);
Y = umap_train.embedding_;        

% % udpate
% delThis = {'embedding_','graph_','graph_dists_',};
% for ii=1:numel(delThis)
%     try
%         umap_train = rmfield(umap_train,delThis{ii});
%     catch
%         fprintf('no umap field: %s\n',delThis{ii})
%     end
% end
%     procInfo.umap.train.umap = umap_train;
%     procInfo.umap.train.savepath = savepath;
%     procInfo.umap.train.name_in = name_in;
% 
% save(sname_procInfo,'-struct','procInfo')
