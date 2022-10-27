datadir = fileparts(anadir);

% prep
nsess = numel(datasets);
nunit = numel(SDF);

nFrame = [];
for ii=1:numel(datasets)
    % load info
    src = [datadir '/' datasets(ii).name '/info.mat'];
    tmp = load(src);
    
    nf = diff(tmp.info.firstLastFrame) ./ 30000 * 30 + 1;
    nFrame(ii) = nf;
end

% modules
tmp = squeeze(res_mod.obs.modularity(1,:,:));
[~,imx] = max(tmp,[],2);
nModules = res_mod.Cutoffs(imx)';

% duration of modules
imod = find(diff(C)~=0);
mod_dur = [imod(1); diff(imod)] ./ 30;
iC_mod = [C(1); C(imod+1)];
id2 = [idat(1); idat(imod)];

n = histcounts2(iC_mod,id2);
m = accumarray([iC_mod,id2],1);

[mu,se] = avganderror(nModules,'mean');
[mud,sed] = avganderror(mod_dur,'mean');
    
% session summaries
fprintf('==========================================\n\t\tsummary: %s\n',monk)

fprintf('# sessions = %d\nmean nFrame = %d\n# frames range = [%d %d]\ntotal # frames = %d\n',...
    nsess,mean(nFrame),[min(nFrame) max(nFrame)],sum(nFrame))

fprintf('\n# actions = %d\nmean action duration = %.3g + %.3g\nmean # modules = %.3g + %.3g\n',...
    max(C),mu,se,mud,sed)

fprintf('\n# cells = %d\n',numel(SDF))

fprintf('==========================================\n',monk)
