%%settings
act_lim = [4 Inf];

%% prep
DGS = cellfun(@(x) x.dasgupta_score,{res_mod_collapse.obs},'un',0);
DGS = cat(1,DGS{:});

DGSr = cellfun(@(x) x.dasgupta_score,{res_mod_collapse.rand},'un',0);
DGSr = cat(1,DGSr{:});

Q = cellfun(@(x) max(x.modularity,[],3),{res_mod_collapse.obs},'un',0);
Q = cat(1,Q{:});

Qr = cellfun(@(x) max(x.modularity,[],3),{res_mod_collapse.rand},'un',0);
Qr = cat(1,Qr{:});
Qr = squeeze(Qr);

naction = [res_mod_collapse.naction];

% downsample
sel = naction >= act_lim(1) & naction <= act_lim(2);
naction(~sel) = [];
DGS(~sel,:,:) = [];
DGSr(~sel,:,:) = [];
Q(~sel,:,:) = [];
Qr(~sel,:,:) = [];

% where does modularity stop changing?
bad = sum(~isnan(Q),2) < 2;

%% start figure
figure
nr = 1; nc = 2;
set_bigfig(gcf,[0.5 0.3])

%% mean modularity as a function of cluster granularity
Qr2 = nanmean(Qr,3);

[mu,se] = avganderror(Q,'mean',2);
[mur,ser] = avganderror(Qr2,'mean',2);

p = sum(nanmean(Qr,2) > mu,3) ./ size(Qr,3);
p = bonf_holm(p);
mup = ones(size(p))* max([mu+se;mur+ser]) *1.05;
mup(p>0.05) = nan;

ns = 1;
subplot(nr,nc,ns)

h=[];
htmp = shadedErrorBar(naction,mu,se,{'color',ones(1,3)*0});
h(1) = htmp.mainLine;
hold all
htmp = shadedErrorBar(naction,mur,ser,{'color',ones(1,3)*0.5});
h(2) = htmp.mainLine;
plot(naction,mup,'k.','markersize',10)

axis square
legend(h,{'obs','rand'},'location','southeast')
title('modularity as a function of embedding granularity')
xlabel('# actions')
ylabel('mean modularity')


%% mean dasgupta as a function of cluster granularity

[mu,se] = avganderror(DGS,'mean',2);
[mur,ser] = avganderror(nanmean(DGSr,3),'mean',2);

p = sum(nanmean(DGSr,2) > mu,3) ./ size(DGSr,3);
p = bonf_holm(p);
mup = ones(size(p))* max([mu+se;mur+ser]) *1.05;
mup(p>0.05) = nan;

ns = 2;
subplot(nr,nc,ns)

h=[];
htmp = shadedErrorBar(naction,mu,se,{'color',ones(1,3)*0});
h(1) = htmp.mainLine;
hold all
htmp = shadedErrorBar(naction,mur,ser,{'color',ones(1,3)*0.5});
h(2) = htmp.mainLine;
plot(naction,mup,'k.','markersize',10)

axis square
legend(h,{'obs','rand'},'location','southeast')
title('hierarchy as a function of embedding granularity')
xlabel('# actions')
ylabel('mean dasgupta score')


%% save figure
sname = [figdir '/embedding_collapse_modHier.pdf'];
save2pdf(sname,gcf)





