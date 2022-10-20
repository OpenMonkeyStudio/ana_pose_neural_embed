% NB: run ana_embedding first

% settings
verbose = 1;

% paths
[~,~,~,rpath] = set_pose_paths(0);

%anadir = '/mnt/scratch/BV_embed/embed_rhesus_jointAngle';
anadir = '/mnt/scratch/BV_embed/embed_rhesus_jointAngle_vcom_pcom_preAlignPose_pca';
if ~exist(anadir); mkdir(anadir); end

tmp = [get_code_path() '/bhv_cluster/matlab/R/call_sigclust.R'];
[funcpath, func] = fileparts(tmp);

% get all leaves
tree_unfurled = unfurl_tree(tree,nstate);
save('tree_unfurled.mat','tree_unfurled')

% prep data
% rng(42)
% p = 0.1;
% sel = false(size(C));
% for ic=1:nstate
%     s = find(C==ic);
%     nsmp = ceil(sum(s)*
%     nsmp = size(X_notran,1);
%     %sel = randperm(nsmp,ceil(nsmp*0.05));
%     sel = 1:nsmp;
% end
sel = true(size(C));

% save
dat = [];
dat.T = tree_unfurled;
dat.T_size = size(tree_unfurled);
dat.X = Y(sel,:);
dat.label = C(sel);
dat.nrand = 200;
dat.nsim = 100; 
dat.pItem = 1;
dat.maxItem = 5000;

tmpname = [anadir '/sigclust_data'];
name_in = [tmpname '_in.mat'];
name_out = [tmpname '_out.mat'];

save(name_in,'-struct','dat')

% call the func
tic
commandStr = sprintf('cd %s; %s %s.R "%s"',funcpath,rpath,func,tmpname);

if verbose
    [status,result] = system(commandStr,'-echo');
else
    [status,result] = system(commandStr);
end

% load back
out = load(name_out);
toc