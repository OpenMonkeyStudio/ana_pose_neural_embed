function out = plot_anova(K2,iG,cfg)
% out = plot_anova(K2,iG,cfg)

% checks
cfg = checkfield(cfg,'groups',cellfun(@num2str,num2cell(unique(iG)),'un',0));
cfg = checkfield(cfg,'ugroups',unique(iG));
cfg = checkfield(cfg,'ylabel','y');
cfg = checkfield(cfg,'title','');
cfg = checkfield(cfg,'dobar',1);
cfg = checkfield(cfg,'func',@mean);

% get means
func = cfg.func;

%xx = 1:numel(cfg.groups);
xx = cfg.ugroups;
mu = grpstats(K2, iG, {func});
se = grpstats(K2, iG,{@(x) nanstd(bootstrp(200,func,x))});
[p, atab] = anovan(K2,iG,'display','off');

% plot mean per area
if cfg.dobar
    hb = barwitherr(se,xx,mu);
else
    hb = errorbar(xx,mu,se);
end

% finish
if ~isempty(cfg.title)
    cfg.title = sprintf('%s\n',cfg.title );
end
str = sprintf('%sAnova F=%.3g, p=%.3g',...
    cfg.title,atab{2,6},atab{2,7});
title(str)
ylabel(cfg.ylabel)

axis square
set(gca,'xtick',xx,'xticklabel',cfg.groups);

% output
out = [];
out.ax = gca;
out.hbar = hb;
out.mean = mu;
out.se = se;
out.xx = xx;
out.p = p;
out.atab = atab;
