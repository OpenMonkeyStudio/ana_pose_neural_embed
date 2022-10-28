function ana_mod_hierarchy(cfg,res_mod)
% ana_mod_hierarchy(cfg,res_mod)

% checks
cfg = checkfield(cfg,'anadir','needit');

cfg = checkfield(cfg,'example_modularity_id','');
cfg = checkfield(cfg,'example_hierarchy_id','');

cfg = checkfield(cfg,'plot_modularity_mean',1);
cfg = checkfield(cfg,'plot_modularity_hist',1);
cfg = checkfield(cfg,'plot_hierarchy_hist',1);

% paths
anadir = cfg.anadir;

figdir = [anadir '/Figures'];
if ~exist(figdir); mkdir(figdir); end

mpath = [anadir '/modularity_test'];
if ~exist(mpath); mkdir(mpath); end

% flags
saveFig = 1;

plotExampleModularity = ~isempty(cfg.example_modularity_id);
plotExampleHierarchy = ~isempty(cfg.example_hierarchy_id);

% prep
nstate = numel(res_mod.ignoredStates{1,1});
nrand = size(res_mod.rand.modularity,5);
lags = res_mod.trans_lags;
Cutoffs = res_mod.Cutoffs;

Qh = res_mod.obs.modularity;
Qhr = res_mod.rand.modularity;

Q = max(Qh,[],3);
Qr = squeeze(max(Qhr,[],3));

DGS = res_mod.obs.dasgupta_score;
DGSr = res_mod.rand.dasgupta_score;

%% example modularity osrting
if plotExampleModularity
    ilag = 1;
    IDs = cfg.example_modularity_id;

    fprintf('plotting transition matrices: ')
    for id1=1:numel(IDs)
        id = IDs(id1);
        fprintf('%g,',id)

        % start figure
        figure
        nr = 1; nc = 3;
        set_bigfig(gcf,[0.7 0.4]);

        % prep transition 
        m = squeeze(res_mod.obs.modularity(ilag,id,:));
        [mx,imx] = max(m);
        lab = squeeze(res_mod.obs.labels_cut(ilag,id,imx,:));
        good = ~isnan(lab);

        tmp = res_mod.obs.po{ilag,id};
        po = nan(nstate,nstate);
        po(good,good) = tmp;

        % plot orig
        ns = 1;
        subplot(nr,nc,ns)
        out1 = plot_sorted_transition(po,lab,0);
        title('orig')

        % plot modularity
        ns = 2;
        subplot(nr,nc,ns)
        plot(res_mod.Cutoffs,m,'k.-','linewidth',2,'markersize',10)
        hold all
        plot(res_mod.Cutoffs(imx),m(imx),'ro','markersize',10)
        s = sprintf('mod=%.3g, n=%g',mx,Cutoffs(imx));
        text(res_mod.Cutoffs(imx),m(imx)*1.05,s,'color','r')

        fn = find(isnan(m),1)-1;
        set(gca,'xlim',[res_mod.Cutoffs(1)-1, res_mod.Cutoffs(fn)+1])
        axis square
        xlabel('nclust')
        ylabel('Modularity')

        % plot sorted
        ns = 3;
        subplot(nr,nc,ns)
        out2 = plot_sorted_transition(po,lab);
        title('sorted')

        if saveFig
            figdir2 = [figdir '/modularity_examples'];
            if ~exist(figdir2); mkdir(figdir2); end
            sname = sprintf('%s/modularity_id%g',figdir2,id);
            save2pdf(sname,gcf)
        end
    end

        
end

%% example hierarchy
if plotExampleHierarchy
    isVis = 1;
    ilag = 1;

    IDs = cfg.example_hierarchy_id;

    fprintf('plotting caldograms: ')
    for id1=1:numel(IDs)
        id = IDs(id1);
        fprintf('%g,',id)

        % get data
        T = res_mod.obs.dendrogram{ilag,id};

        m = squeeze(res_mod.obs.modularity(ilag,id,:));
        [mx,imx] = max(m);
        lab = squeeze(res_mod.obs.labels_cut(ilag,id,imx,:));
        bad = isnan(lab);
        lab(bad) = [];

        % plot
        %https://www.mathworks.com/matlabcentral/answers/324932-dendrogram-with-custom-colouring
        nModule   = max(lab);
        map = [find(~bad), [1:numel(lab)]', lab];

        out = plot_cladogram(T,[],map,nModule,[],isVis,0);
        
        % allow for log spacing, get rid of 0s
        y = get(out.hden,'ydata');
        y = cat(1,y{:});
        mn = min(y(y>0));

        y(y==0) = mn *0.8;

        for ih=1:numel(out.hden)
            set(out.hden(ih),'ydata',y(ih,:));
        end

        axden = out.ax_den;
        set(axden,'yscale','log')
        set(axden,'TickLength',[0.005 0.002])

        tstr = sprintf('dataset: %g, nModule=%g',id,nModule);
        title(axden,tstr)

        % save
        if saveFig
            figdir2 = [figdir '/cladogram_module'];
            if ~exist(figdir2); mkdir(figdir2); end
            sname = sprintf('%s/clado_id%g',figdir2,id);
            save2pdf(sname,gcf)
        end
    end
    fprintf('\n')
end

%% plot mean modulatiry
if cfg.plot_modularity_mean
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
    set_bigfig(gcf,[0.4 0.4])

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
if cfg.plot_modularity_hist

    % start
    figure
    nr = 1; nc = 2;
    set_bigfig(gcf,[0.7 0.5])

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



%% hiearchyq uality scores
if cfg.plot_hierarchy_hist
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
    set_bigfig(gcf,[0.4 0.4])

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




