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

% duration of actions
iaction = find(diff(C)~=0);
act_dur = [iaction(1); diff(iaction)] ./ 30;
iC_act = [C(1); C(iaction+1)];
id2 = [idat(1); idat(iaction)];

n = histcounts2(iC_act,id2);
m = accumarray([iC_act,id2],1);

[mu,se] = avganderror(nModules,'mean');
[mud,sed] = avganderror(act_dur,'mean');
st = nanstd(act_dur);

% session summaries
fprintf('==========================================\n\t\tsummary: %s\n',monk)

fprintf('# sessions = %d\nmean nFrame = %d\n# frames range = [%d %d]\ntotal # frames = %d\n',...
    nsess,mean(nFrame),[min(nFrame) max(nFrame)],sum(nFrame))

fprintf('\n# actions = %d\nmean # modules = %.3g + %.3g SE\nmean action duration = %.3g + %.3gSE, %.3g SD\n',...
    max(C),mu,se,mud,sed,st)

fprintf('\n# cells = %d\n',numel(SDF))

fprintf('==========================================\n',monk)

% plot mean duration
s = accumarray([iC_act,id2],[0; act_dur]);
nd = accumarray([iC_act,id2],1);
m = s ./ nd;
[mua,sea] = avganderror(m,'mean',2);

figure
hb = barwitherr(sea,mua,'facecolor',ones(1,3)*0.5);
xlabel('action')
ylabel('mean duration (s)')
title(['mean duration of each action, monk=' monk])

sname = [figdir '/mean_action_durations_' monk '.pdf'];
save2pdf(sname,gcf)


foo=1;
