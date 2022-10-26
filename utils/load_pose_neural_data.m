[parentdir,jsondir,pyenvpath,rpath,binpath,codepath,ephyspath] = set_pose_paths(0);

%% settings

% neural
useSDF_resid = 1;
forceAreaUpdate = 0;
loadAreas = {'ACC','VLPFC','DLPFC','OFC','SMA','PM'};
areaOrder = {'PM','SMA','DLPFC','VLPFC','ACC','OFC'};

% useful
fs_frame = 30;
fs_neural = 1000;
fs = fs_frame;

%% load embedding
fprintf('loading embedding testing data...\n')

% load orig data
fprintf('\t dataset info ...\n')
tmp = {'idat' 'com' 'frame' 'labels' 'datasets'};
load([anadir '/info.mat'],tmp{:})

% load cluster results
% - assumes has already been cleaned
fprintf('\t cluster...\n')
cluster_test_orig = load([anadir '/cluster_test.mat']);
try, cluster_train_orig = load([anadir '/cluster_train.mat']);catch; cluster_train_orig=[]; end

C = cluster_test_orig.clabels;
cluster_train = cluster_test_orig;
stateMap = [1:max(C), 1:max(C)];

nstate = max(C);

%% try loading modularity, if it exists
mpath = [anadir '/modularity_test'];
sname = [mpath '/modularity_train.mat'];

if exist(sname)
    fprintf('loading modularity data... \n')
    res_mod = load(sname);
end
    
%% load neural data
sdfInfo = load([anadir '/sdfInfo.mat']);
evals(sdfInfo)
[SDF,sdfnames] = load_back_sdf(sdfpath_data,useSDF_resid,{},{},loadAreas,1,forceAreaUpdate);

% collapse areas for analysis
areas = [SDF.area]';
areas = collapse_areas(areas);
[uarea,~,iarea] = unique(areas);

% remap for plotting
[~,iarea2] = ismember(areas,areaOrder);
tmp = [iarea, iarea2];
tmp = unique(tmp,'rows');
iarea = changem(iarea,tmp(:,2),tmp(:,1));
uarea = areaOrder;
