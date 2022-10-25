% paths
figdir = [anadir '/Figures'];
if ~exist(figdir); mkdir(figdir); end

mpath = [anadir '/modularity_test'];
if ~exist(mpath); mkdir(mpath); end

% flags
saveFig = 1;

plotMeanModularity = 1;
plotModuleHist = 1;
plotDasguptaHist = 1;

% prep
nrand = size(res_mod.rand.modularity,5);
lags = res_mod.trans_lags;
Cutoffs = res_mod.Cutoffs;

Qh = res_mod.obs.modularity;
Qhr = res_mod.rand.modularity;

Q = max(Qh,[],3);
Qr = squeeze(max(Qhr,[],3));

DGS = res_mod.obs.dasgupta_score;
DGSr = res_mod.rand.dasgupta_score;

%% plot mean modulatiry
if plotMeanModularity
    % get means
    [mu,se] = avganderror(Q,'mean',2);

    tmpq = nanmean(Qr,3);
    [mur,ser] = avganderror(tmpq,'mean',2);

    % significance
    tmpmu = nanmean(Qr,2);
    p = (sum(tmpmu >= mu,3)+1) ./ (nrand+1);
    p = bonf_holm(p);

    mup = ones(size(p)) * max(mu+se)*1.02;
    mup(p>0.05) = nan;
    
    % plot
    figure
    set_bigfig(gcf,[0.25 0.35])

    h = [];
    htmp = shadedErrorBar(lags,mu,se,{'r.-'});
    h(1) = htmp.mainLine;
    hold all
    htmp = shadedErrorBar(lags,mur,ser,{'.-','color',ones(1,3)*0.5});
    h(2) = htmp.mainLine;
    plot(lags,mup,'k.-')

    xlabel('# transitions')
    ylabel('mean modularity')
    legend(h,{'obs','rand'})
    set(gca,'xscale','log','xlim',[lags(1), lags(end)])
    axis square

    if saveFig
        sname = [figdir '/mean_modularity'];
        save2pdf(sname,gcf)
    end
end


%% modularity histogram
if plotModuleHist

    % start
    figure
    nr = 1; nc = 2;
    set_bigfig(gcf,[0.35 0.25])

    % ------------------------------------
    % histogram for lag==1
    ilag = 1;

    % get means
    q = Q(ilag,:);
    qr = Qr(ilag,:,:);

    % significance
    pi = sum(qr > q,3) ./ (size(qr,3)+1);

    [mu,se] = avganderror(q,'mean');
    tmpmu = nanmean(qr,2);
    p = (sum(tmpmu >= mu,3)+1) ./ (nrand+1);
    qr = nanmean(qr,3);

    subplot(nr,nc,1)
    ho = histogram(q,'normalization','probability');
    set(ho,'facecolor','k','edgecolor','none')
    hold all
    hr = histogram(qr,'normalization','probability');
    set(hr,'facecolor',ones(1,3)*0.8,'edgecolor','none')

    legend([ho,hr],{'obs','rand'})

    str = sprintf('Modularity Scores\npop-p=%.3g\n num sign sets: %g/%g',...
        p,sum(pi<0.05),numel(pi));
    title(str)
    xlabel('Modularity Score')
    ylabel('Proportion')
    axis square

    % ------------------------------------
    % mean across modules
    q = squeeze(Qh(ilag,:,:));
    qr = squeeze(Qhr(ilag,:,:,:));

    [mu,se] = avganderror(q,'mean');
    [mur,ser] = avganderror(nanmean(qr,3),'mean');

    tmp = squeeze(nanmean(qr))';
    p = sum(tmp > mu) ./ size(tmp,1);
    mup = ones(size(mu))*max(mu+se)*1.05;
    mup(p>0.05) = nan;

    subplot(nr,nc,2)
    %xx = 2:nstate;
    xx = Cutoffs;
    shadedErrorBar(xx,mu,se,{'color','k'})
    hold all
    shadedErrorBar(xx,mur,ser,{'color',ones(1,3)*0.8})
    plot(xx,mup,'k.','markersize',5)

    set(gca,'ylim',[0 max(get(gca,'ylim'))])

    title('Modularity Scores by transition lag')
    xlabel('# modules')
    ylabel('Mean Modularity Score')
    axis square

    % save

    if saveFig
        sname = [figdir '/hist_modularity_lag' num2str(lags(ilag))];
        save2pdf(sname,gcf)
    end

    foo=1;
end



%%
if plotDasguptaHist
    % ------------------------------------
    % histogram for lag==1
    ilag = 1;

    % get means
    q = squeeze(DGS(ilag,:))';
    qr = squeeze(DGSr(ilag,:,:));

    % individual significance
    pi = sum(qr > q,2) ./ size(qr,2);

    % significance
    p = (sum(nanmean(qr) >= mean(q),2)+1) ./ (size(qr,2)+1);
    qr = nanmean(qr,2);

    figure
    set_bigfig(gcf,[0.3 0.3])

    ho = histogram(q,'normalization','probability');
    set(ho,'facecolor','k','edgecolor','none')
    hold all
    hr = histogram(qr,'normalization','probability');
    set(hr,'facecolor',ones(1,3)*0.8,'edgecolor','none')

    legend([ho,hr],{'obs','rand'},'location','eastoutside')

    str = sprintf('Dasgupta Scores\npop-p=%.3g\n num sign sets: %g/%g\n',...
        p,sum(pi<0.05),numel(pi));
    title(str)
    xlabel('Dasgupta Score')
    ylabel('Proportion')
    axis square

    if saveFig
        sname = [figdir '/hist_dasgupta_lag' num2str(lags(ilag))];
        save2pdf(sname,gcf)
    end
end




