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

sname = [anadir '/sdfInfo.mat'];
tmp = load(sname);
sdfpath = [anadir '/' tmp.sdfname];
sdfpath_data = [anadir '/' tmp.sdfname '/sdf'];

fprintf('loading data:\n%s\n',anadir)
tic

%% load embedding
fprintf('loading embedding testing data...\n')

% load orig data
fprintf('\t dataset info ...\n')
tmp = {'idat' 'com' 'frame' 'labels' 'datasets','idx_train'};
load([anadir '/info.mat'],tmp{:})

% load cluster results
% - assumes has already been cleaned
fprintf('\t cluster...\n')
cluster_train = load([anadir '/cluster_train.mat']);
cluster_test = load([anadir '/cluster_test.mat']);

C = cluster_test.clabels;
stateMap = [1:max(C), 1:max(C)];

nstate = max(C);

%% try loading modularity, if it exists
mpath = [anadir '/modularity_test'];

sname = [mpath '/modularity_train.mat'];
if exist(sname,'file')
    fprintf('loading modularity data... \n')
    res_mod = load(sname);
end
    
mpath2 = [anadir '/modularity_test_collapse'];
dd= dir([mpath2 '/naction*']);

if numel(dd) > 0
    res_mod_collapse = [];
    for ii=1:numel(dd)
        name = dd(ii).name;
        n = str2double(name(8:end));
        tmp = load([mpath2 '/' name '/modularity_train.mat']);
        tmp.naction = n;
        res_mod_collapse = cat(1,res_mod_collapse,tmp);
    end
    
    [~,is] = sort([res_mod_collapse.naction]);
    res_mod_collapse = res_mod_collapse(is);
end

%% load neural data
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

toc