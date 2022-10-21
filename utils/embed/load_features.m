function [X_feat,out] = load_features(featdir,suffix,datasets)
% [X_feat,feat_labels,ifeat,idat_feat] = load_features(featdir,suffix,datasets)

% stuff
nsmp_tot = numel(datasets)*200000;
fn = 0;

out = struct('frame',nan(nsmp_tot,1),'idat',nan(nsmp_tot,1),'com',nan(nsmp_tot,3));

% load
fprintf('loading features, ndata=%g\n%s\n',numel(datasets),featdir)
for ii=1:numel(datasets)
    fprintf('%g,',ii)
    name = datasets(ii).name; 
    sname = [featdir '/' name '_' suffix '.mat'];

    % load
    in = load(sname);
    
    % init
    if ii==1
        X_feat = nan(nsmp_tot,numel(in.feat_labels));
    end

    % append
    st = fn+1;
    fn = fn + size(in.X_feat,1);
    X_feat(st:fn,:) = in.X_feat;
    out.frame(st:fn) = in.frame;
    out.com(st:fn,:) = in.com;
    out.idat(st:fn) = ii;
end
fprintf('\n')

% cull
X_feat(fn+1:end,:) = [];
out.frame(fn+1:end,:) = [];
out.com(fn+1:end,:) = [];
out.idat(fn+1:end,:) = [];

% finish
out.feat_labels = in.feat_labels;
out.ifeat = in.ifeat;
out.datasets = datasets;
out.labels = in.labels;
out.featInfo = in.featInfo;
out.info = in.info;

