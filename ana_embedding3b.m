[parentdir,jsondir,pyenvpath,rpath] = set_pose_paths(0);

% anadir = '/Volumes/SSD_Q/P_embedding/embed_rhesus_jointAngle';
% anadir = '/Volumes/SSD_Q/P_embedding/embed_rhesus_jointAngle_vcom_pcom_minDist_0-1';
% anadir = '/Volumes/SSD_Q/P_embedding/embed_rhesus_jointAngle_vcom_pcom_preAlignPose_midUmap';
%anadir = '/Volumes/SSD_Q/P_embedding/embed_rhesus_jointAngle_vcom_pcom_preAlignPose_pca_v2';
% anadir = '/Volumes/SSD_Q/P_embedding/embed_rhesus_jointAngle_vcom_pcom_preAlignPose';
%anadir = '/Volumes/SSD_Q/P_embedding/embed_rhesus_jointAngle_vcom_pcom_preAlignPose_minUmap';
% anadir = '/Volumes/SSD_Q/P_embedding/embed_rhesus_jointAngle_spec';

%anadir = [parentdir '/embed_rhesus_jointAngle_vcom_pcom_preAlignPose_pca_v2'];
%anadir = [parentdir '/P_embedding/embed_rhesus_jointAngle_vcom_pcom_noAlignPose_pca_v2'];
%anadir = [parentdir '/P_neural_embed/embed_rhesus_jointAngle_vcom_pcom_noAlignPose_pca_v2_test'];

%anadir = [parentdir '/P_neural_embed_slim/embed_rhesus_jointAngle_vcom_pcom_noAlignPose_pca_new_13j'];
%anadir = [parentdir '/P_neural_embed/embed_rhesus_jointAngle_vcom_pcom_noAlignPose_pca_new_13j'];


figdir = [anadir '/Figures'];
if ~exist(figdir); mkdir(figdir); end


% flags
saveFig = 1;

loadData = 1;
    cleanStates = 0;
    trainData = 0;

getPeakPoses = 1;
treeFromPose = 0;
saveFramesToReproject = 0;

getSummary = 1;

plotEmbedding = 1;
plotEigenMonkey = 0;
plotMeanHeight = 0;
plotCladogram = 0;
    useBranchPval = 0;
    plotCladoPoses = 1;
plotMonkeySpecificity = 0;

plotExampleEthogram = 0;
plotPoseVids = 0;

plotExampleSkeleton = 0;
plotSilh = 0;
plotCorrDistance = 0;
plotCorrDistVel = 0;

plotTransition = 0;
plotCorrTranProbDistance = 0;
plotModularityHierarchy = 0;
    calcModularityHierarchy = 1;
plotModularityLag = 0;
    calcModularity = 0;
    loadModularity = 0;
plotTreeness = 0;
plotProbRevisitState = 0;
    doCalculations = 0;
plotSequenceProb = 0;
plotMarkovTimescales = 0;
    doCalculationsMarkov = 1;
plotXCorr = 0;
plotEthoMatch = 0;
plotEmbedSeq = 0;
    plotSeqEmbedding = 1;
    plotSeqMonkeySpecificity = 1;
    plotSeqs = 1;
    plotSeqTree = 0;
plotNNFM = 0;

% settings
fs = 30;
minSmp = 3;
pSmp = 0.001;

if treeFromPose
    dendDist = 'euclidean';
    linkMethod = 'complete';
    dendCutoff = 1.9;
else
    dendDist = 'euclidean';
    %linkMethod = 'ward'; % XX_joint
    %dendCutoff = 12.9;
    linkMethod = 'complete'; % average
    %dendCutoff = 6.5;
    dendCutoff = 9.4;
end

%% load data
cd(anadir)
if loadData
    fprintf('loading...\n')
    
    % orig data, for plotting
    fprintf('\torig data...\n')

    if trainData
        orig_data = load('data_train.mat');
    else
        orig_data = load('data_test.mat');
    end
    
    com = orig_data.com;
    frame = orig_data.frame;
    idat = orig_data.idat;
    %datasets = tmp.datasets;
    labels = orig_data.labels;
    datasets = orig_data.datasets;
    
    if 0
        X_spine = orig_data.X_spine;
    else
        warning('loading input X for plotting...')
        load('X_orig.mat')
        X_spine=X_orig(orig_data.sample_idx,:);
    end
    X_notran = orig_data.X_notran;
    
    % embedding
    fprintf('\tembedding...\n')

    if trainData
        umap_train = load('umap_train.mat');
    else
        umap_train = load('umap_test.mat');
    end
    Y = umap_train.embedding_;
    
    % cluster results
    fprintf('\tcluster...\n')

    if trainData
        cluster_train_orig = load('cluster_train.mat');
    else
        cluster_train_orig = load('cluster_test.mat');
    end
    Lbnds = cluster_train_orig.Lbnds;
    Ld = cluster_train_orig.Ld;
    clabels = cluster_train_orig.clabels;
    outClust_orig = cluster_train_orig.outClust;
end

if cleanStates
    [C,stateMap] = clean_states(clabels,minSmp,pSmp);
    cluster_train = clean_clusterInfo(cluster_train_orig,stateMap);
else
    C = clabels;
    cluster_train = cluster_train_orig;
    stateMap = [1:max(clabels), 1:max(clabels)];
end

Lbnds = cluster_train.Lbnds;
Ld = cluster_train.Ld;
outClust = cluster_train.outClust;

%% useful
istate = find(diff(C)~=0);
C_state = C(istate);

nstate = max(C_state);

% duration
C_dur = [istate(1); diff(istate)];
C_dur_orig = C_dur;

monks = unique(cellfun(@(x) x(1:2),{datasets.name},'un',0));
for im=1:numel(monks)
    selm = ismember(idat(istate),find(strncmp({datasets.name},monks{im},2)));
    tmp = C_dur(selm);
    mu = nanmedian(tmp);
    s = mad(tmp,1) * 1.4826;
    C_dur(selm) = (tmp-mu) ./ s;
end
%s = mad(C_dur) * 1.486;
%s = median(C_dur);
%c = C_dur ./ s;

% cluster mediods and distance
[C_mediod,uclust] = get_cluster_mediods(Y,C);


% poses at peaks in clusters
if getPeakPoses
    fprintf('getting median poses... \n')
    lim = 0.5;
    nk = 100;
    
    pose_mu = [];
    for ic=1:nstate
        % now get the mean pose
        xy = C_mediod(ic,:);

        % closest samples

        if 0 % within limits
            x = Y(:,1);
            y = Y(:,2);

            seldata = (x >= xy(1)-lim & x<= xy(1)+lim) &...
                        (y >= xy(2)-lim & y<= xy(2)+lim);
        else % N closests samples
            seldata = find(C==ic);
            tmpy = Y(seldata,:);
            nk2 = min(nk,ceil(size(tmpy,1)/2));
            k = knnsearch(tmpy,xy,'k',nk2);
            seldata = seldata(k);
        end

        tmp = X_spine(seldata,:);
        mu = nanmedian(tmp,1);

        pose_mu(ic,:) = mu;
    end
end


% get tree
if treeFromPose
    D = pdist(pose_mu,dendDist);
    tree = linkage(D,linkMethod);
    leafOrder = optimalleaforder(tree,D);
    tClust = cluster(tree,'cutoff',dendCutoff,'Criterion','distance');  
else
    D = pdist(C_mediod,dendDist);
    tree = linkage(D,linkMethod);
    leafOrder = optimalleaforder(tree,D);
    tClust = cluster(tree,'cutoff',dendCutoff,'Criterion','distance'); 
end

% rename labels for clarity
% newval = 1:nstate;
% oldval = leafOrder;
% 
% uclust = changem(uclust,newval,oldval);
% %leafOrder = changem(leafOrder,oldval,newval);
% Ld = changem(Ld,newval,oldval);
% C = changem(C,newval,oldval);
% C_state = changem(C_state,newval,oldval);

if saveFramesToReproject
    nimg = 20;
    
    f = frame;
    ds = cellfun(@(x) x(1:end-9), {datasets.name},'un',0);
    id = idat;
    c = C;
    
    % some selection
    sel = true(size(c));
            
    f(~sel) = [];
    c(~sel) = [];
    id(~sel) = [];
    
    % save
    frameInfo = get_frames_to_reproject(anadir,f,ds,id,c,nimg);
end


%% summary of extracted states
if getSummary
    nstate = max(C);
    c = C_dur_orig ./ fs;
    [mu,se] = avganderror(c,'mean');
    
    fprintf('nstate=%g\nmean dur=%.3g + %.3g\n',nstate,mu,se);
end

%% sample pose
if plotExampleSkeleton
    % start from 3rd cluster
    m2 = [-0.0239826811776010,0.134178125090325,0.199021689792069,-0.00107628668585050,0.221793261638662,0.0726643058715109,0,0,0,-0.0973528364028673,-0.0743733530150859,-0.0734143364520356,-0.139233798684831,0.100000000000000,0.109270926370527,0.113654035161082,-0.0808994819618685,-0.0179802659928683,0.162546555480189,0.100000000000000,0.169314324454674,-3.97010637694181e-20,-0.943306549430069,-0.331922812776087,-0.164732604374200,-0.987672242384239,-0.132836270252862,-0.178543230500814,-1.37314772095247,-0.107226409819279,0.172541413423281,-1.06332302876394,-0.144474136861791,0.193112741297576,-1.44354764238012,-0.111480043861582,0.00688789322031905,-1.43963280926461,-0.374445360507436,-0.120000000000000,-0.159983344323271,-0.0283241502152419,0.120000000000000,-0.137302738047027,0.0145439476548120];
    m2(20) = 0.1; % left hand
    m2(43) = 0.12; % left elbow
    m2(14) = 0.1; %right hand
    m2(40) = -0.12; % right elbow

    figure; 
    plot_monkey_slim({m2,labels},[],1);
    set(gca,'visible','off')
    view(-147, 22)

    sname = [figdir '/sample_pose'];
    str = sprintf('print -painters -dsvg %s',sname);
    eval(str)
end

%% example ethogram
if plotExampleEthogram
    figdir2 = [figdir '/ethogram'];
    if ~exist(figdir2); mkdir(figdir2); end
    
    nsmp = 1000;
    st = 1000;
    for id=1:max(idat)
        sel = idat==id;
        c = C(sel);
        idx = st:st+1000;
        c = c(idx);
                
        % plot
        figure
        for ic=1:nstate
            xx = [idx(find(c==ic))-idx(1)] ./ fs;
            yy = ones(size(xx)) * ic;
            plot(xx,yy,'.','markersize',10)
            hold all
        end
        
        grid on
        set(gca,'ylim',[0 nstate+1])
        
        ylabel('pose')
        xlabel('time')
        
        % save
        sname = [figdir2 '/etho_' num2str(id)];
        save2pdf(sname,gcf)
        close(gcf)
    end
end

%% monkey specificity
if plotMonkeySpecificity
    nboot = 200;
    
    % start figure
    figure('name','residency')
    set_bigfig(gcf,[0.7 0.6])
    nr = 3; nc = 1;
    
    % calculations
    nclust = accumarray(C,1);
    [n,xe,ye] = histcounts2(C,idat,[max(C) max(idat)]);
    p = n ./ nclust;
    
    nclust2 = accumarray(C,1);
    p2 = nclust2 ./ sum(nclust2(:));
    
    %p = p ./ accumarray(idat,1)';
    th = 1/max(idat);
    b = mean(abs(p-th),2) * 100;
    b = sum(p2.*b);
        
    cols = get_safe_colors(0,[1 2]);
    monks = {'yo','ca'};
    monk = cellfun(@(x) x(1:2), {datasets.name},'un',0);
    [~,~,nmonk] = unique(monk);
    nmonk = accumarray(nmonk,1);
    
    % monkey specificity
    subplot(nr,nc,1)
    for id=1:max(idat)
        im = ismember(monks,monk(id));
        tmpp = p(:,id);
        plot(1:max(C),tmpp,'.-','markersize',10,'color',cols(im,:))
        hold all
    end
    
    plotcueline('y',th,'k:')
    str = sprintf('probability of residency by monk\nweighted bias=%.3g %%',b);
    title(str)
    
    legstr = cellfun(@(x) x(1:2), {datasets.name},'un',0);
    legend(legstr,'location','eastoutside')
    pause(0.1)
    pos1 = get(gca,'position');
    
    % monkey bias    
    imonk1 = find(strcmp(monk,monks{1}));
    imonk2 = find(strcmp(monk,monks{2}));

    B = [];
    for ib=1:nboot
        sel1 = randi(numel(imonk1),nmonk(1),1);
        sel2 = randi(numel(imonk2),nmonk(2),1);
        
        p1 = p(:,sel1);
        p2 = p(:,sel2);
        
        d1 = median(abs(p1-th),2) * 100;
        d2 = median(abs(p2-th),2) * 100;
        
        B(:,ib) = d2 - d1;
        foo=1;
    end
    mu = nanmean(B,2);
    se = nanstd(B,[],2);
    
    subplot(nr,nc,2)
    shadedErrorBar(1:nstate,mu,se,{'k.-','markersize',5})
        
    pos2 = get(gca,'position');
    pos2 = [pos2(1:2) pos1(3) pos2(4)];
    set(gca,'position',pos2)
    
    % residency
    subplot(nr,nc,3)

    plot(p2,'.-','markersize',10)
    title('residency')
    pos2 = get(gca,'position');
    pos2 = [pos2(1:2) pos1(3) pos2(4)];
    set(gca,'position',pos2)
    
    
    % save
    if saveFig
        sname = [figdir '/pose_residency_specificty'];
        save2pdf(sname,gcf)
    end
end

%% ----------------------------------------------------------------------
% plot dendrogram
if plotCladogram
    disp('cladogram...')
    
    lim = 0.5;

    poses = pose_mu;
    %m = C_mediod;
    m = pose_mu;
    
    % start figure 
    figure('name','cladogram')
    if plotCladoPoses
        nr = 4; nc = ceil(nstate/(nr-1));
        set_bigfig(gcf,[1 0.5])
    else
        nr = 1; nc = 1;
        set_bigfig(gcf,[0.5 0.3])
    end
    
    % plot claudogram
    ns = 1:nc;
    subplot(nr,nc,ns)
    
    [hden,T,tTree] = dendrogram(tree,nstate,'reorder',leafOrder,'colorthreshold',dendCutoff);
    %[hden,~,tTree] = dendrogram(tree,'colorthreshold',dendCutoff);
    hold all
    
    % pvalues?
    if useBranchPval
        ins = load([anadir '/sigclust_data_out.mat']);
        P = ins.P;
        P = P*numel(P); % bonferoni correction
        
        for ii=1:size(tree,1)    
            p = P(ii);
            x = get(hden(ii),'xdata');
            y = get(hden(ii),'ydata');
            x = mean(x);
            y = max(y);
            
            if p>0.05
                plot(x,y,'ko','markersize',5)
            else
                plot(x,y,'k.','markersize',10)
            end
        end
    end
    
    % plot poses
    if plotCladoPoses
        % map leaves to proper line handle
        tmpx = get(hden,'XData');
        tmpx = cat(1,tmpx{:});
        tmpy = get(hden,'YData');
        tmpy = cat(1,tmpy{:});
        sel = tmpy==0; %terminal leaves
        
        hleaf = nan(size(tTree));
        for ii=1:size(tmpx,1)
            for jj=1:size(tmpx,2)
                
                % if its a terminal node...
                if sel(ii,jj)==1 
                    it = tmpx(ii,jj);
                    hleaf(it) = hden(ii);
                end
            end
        end
        
        % map leaf node to threshold based cluster
        cols_in = get(hleaf,'color');
        cols_in = cat(1,cols_in{:});
        
        % recolor?
        if 0
            cols_out = get_safe_colors(0);
            ucol = unique(cols_in,'rows');
            
            for ii=1:numel(unique(tClust))
                sel = ismember(cols_in,ucol(ii,:),'rows');
                idx = find(ismember(leafOrder,find(sel)));
                
                c = cols_out(ii,:);
                set(hden(sel),'color',c)
            end
        end        
        
        % plot
        for it=1:numel(tTree)
            ip = tTree(it);
            ns = nc + it;
            subplot(nr,nc,ns)

            m = poses(ip,:);
            m = m + ip;
            plot_monkey({m,labels},[],1);

            set(gca,'xtick',[],'ytick',[],'ztick',[])

            s = sprintf('%g',ip);
            title(s)
            xlabel('')
            ylabel('')
            zlabel('')

            if 1
                box on
                c = cols_in(it,:);
                set(gca,'Xcolor',c,'ycolor',c,'zcolor',c)
            end
            foo=1;
        end
    end
    
    % save
    if saveFig
        sname = [figdir '/cladogram'];
        save2pdf(sname,gcf)
    end
end



%% plot embedding
if plotEmbedding
    hf = figure('name','embedding desnity');
    hEmbed = plot_embedding(outClust,Lbnds,Ld);
    
    % explore_embedding_space_continuous(hf,Y,X_spine,labels);
    
    if saveFig
        sname = [figdir '/embedding_density'];
        save2pdf(sname,gcf)
    end
end

%% asses cluster cohesion
if plotSilh
    nboot = 100;
    nsmp = ceil(size(Y,1) * 0.001);
    
    G = [];
    fprintf('silhouette boostrap')
    for ib=1:nboot
        dotdotdot(ib,0.1,nboot)
        idx = randperm(size(Y,1),nsmp);
        y2 = Y(idx,:);
        c = C(idx);
        
        g = silhouette(y2,c);
        G(ib) = mean(g);
    end
end

%% correlate distance in embedding with pose distance
if plotCorrDistance
    
    ndat = numel(C);
    nsmp = 5000;
    nboot = 100;
    
    R = [];
    Rr = [];
    P = [];
    for ib=1:nboot
        idx = randperm(ndat,nsmp);
        
        % pose distance
        x = X_spine(idx,:);
        dx = pdist(x);
        
        % embedding distance
        y = Y(idx,:);
        dy = pdist(y);
        
        % correlate
        [r,p] = corr(dx',dy');
        R(ib) = r;
        P(ib) = p;
        foo=1;
        
        % random correlation
        dyr = dy(randperm(numel(dy)));
        [rr,~] = corr(dx',dyr');
        RR(ib) = rr;
    end
    
    % report
    mu = nanmean(R);
    p = sum(RR > R) ./ nboot;
    
    fprintf('correlate embed vs pose distance: R=%.3g, p=%.3g\n',mu,p);
    
    foo=1;
end

%% correlate distance in embedding with velocity
if plotCorrDistVel
    % get velocity in embedding space
    fs = 30;
        
    V = nan(size(idat));
    for id=1:max(idat)
        sel = idat==id;
        tmpc = Y(sel,:);
        %tmpc = com(sel,:);
        tmpf = frame(sel);
        
        d = diff(tmpc,[],1);
        d = sqrt(sum(d.^2,2));

        dt = diff(tmpf) ./ fs;
        if isrow(dt); dt=dt'; end

        v = [0; d ./ dt];
        V(sel) = v;
    end
    
    % for each data point, get the mean velocity
    xv = [min(outClust.xv) - 0.5; outClust.xv];
    yv = [min(outClust.yv) - 0.5; outClust.yv];
    
    % esimate embedding density
    %[~,~,~,binx,biny] = histcounts2(Y(:,1),Y(:,2),xv,yv);
    
    % normalize embedding density per cluster
    dens = outClust.dens2;
    %Ld = outClust.Ld;
    pembed = nan([size(dens) max(C)]);
    for ic=1:max(C)
        sel = Ld==ic;
        tmp = dens;
        tmp(~sel) = nan;
        tmp = tmp ./ max(tmp(:));
        %tmp = tmp ./ nansum(tmp(:));
        pembed(:,:,ic) = tmp;
    end
    %nembed = accumarray([binx biny clabels],1);
    %pembed = nembed ./ sum(sum(nembed,1),2);
    %pembed = pembed ./ max(max(pembed,[],1),[],2);
    %pembed = nembed ./ max(max(nembed,[],1),[],2);
    
    % get mean velocity, per cluster
    nvel = accumarray([binx biny C],1);
    embed_vel = accumarray([binx biny C],V);
    embed_vel(nvel==0) = nan;
    embed_vel = embed_vel ./ nvel;
    
    % normalize each cluster by its peak velocity
    mx = max(max(embed_vel,[],1),[],2);
    embed_vel2 = embed_vel;
    embed_vel2  = embed_vel2 ./ mx;
    
    % correlate (normalized) vel with (normalized) embedding density
    tmp=repmat(Ld,1,1,38); tmp = tmp(:);
    x = pembed(:);
    y = embed_vel2(:);
    bad = isnan(x) | isnan(y) | tmp==0;
    %bad = bad | y>0.1;
    x(bad) = [];
    y(bad) = [];
    
    corrtype = 'spearman';
    [r,p] = corr(x,y,'type',corrtype);
    
    str = sprintf('correlate norm embedding denisty with norm embedVel\n%s R=%.3g, p=%.3g',...
        corrtype,r,p);
    disp(str)
    
    % plot correlation
    figure('name','corr-vel-dens')
    
    hs = scatter(x,y);
    set(hs,'markerfacecolor',ones(1,3)*0.5,'markeredgecolor','none','MarkerFaceAlpha',0.5)
    hl = lsline;
    set(hl,'color','k')
    
    title(str)
    xlabel('embed dens')
    ylabel('embed velocity')
    
    if saveFig
        sname = [figdir '/corr_vel_dens'];
        save2pdf(sname,gcf)
    end
    foo=1;
end

%% eigenmonkey
if plotEigenMonkey
    % plot
    figure('name','all poses')
    [nr,nc] = subplot_ratio(numel(uclust));
    
    for ip=1:numel(uclust)
        ns = ip;
        subplot(nr,nc,ns)

        m = pose_mu(ip,:);
        plot_monkey_slim({m,labels},[],1);

        set(gca,'xtick',[],'ytick',[],'ztick',[])
        set(gca,'visible','off')

        s = sprintf('%g',ip);
        %title(s)
        pt = [min(get(gca,'xlim')), max(get(gca,'ylim')), min(get(gca,'zlim'))];
        text(pt(1),pt(2),pt(3),s)
        xlabel('')
        ylabel('')
        zlabel('')
        foo=1;
    end
    tightfig
    set_bigfig(gcf,[0.85 0.85])

    if saveFig
        sname = [figdir '/all_poses'];
        str = sprintf('print -painters -dsvg %s',sname);
        eval(str)
        save2pdf(sname,gcf)
    end
end

%% plot average height, and position in cage, for each pose
if plotMeanHeight
    
end


%% transition matrix
if plotTransition
    % get probabilities
    po   = calculate_cond_trans_prob(C_state, [],nstate,1);
    
    % reorder
    ts = tClust(leafOrder);
    uts = unique(ts);
    
    po = po(:,leafOrder);
    po = po(leafOrder,:);
    
    % probability of within vs between transitions
    P = [];
    for it=1:max(ts)
        sel = ts==uts(it);
        
        % within
        tmp_wt = po;
        tmp_wt(~sel,:) = nan;
        tmp_wt(:,~sel) = nan;
        
        % between
        tmp_bw = po;
        tmp_bw(sel,:) = nan;
        tmp_bw(:,sel) = nan;
        
        % stats
        P(it,1) = nanmean(tmp_wt(:));
        P(it,2) = nanmean(tmp_bw(:));
    end
    d = diff(P,[],2);
    [mu,se] = avganderror(d,'mean');
    [~,p] = ttest(P(:,1),P(:,2));
    
    % plot
    figure('name','transition t+1')
    imagesc(po)
    hold all
    
    % bounding boxes
    st = 1;
    for ii=1:max(ts)
        s = ts(st);
        fn = find(ts==s,1,'last');
        st = st-0.5;
        fn = fn+0.5;
        
        xx = [st fn fn st st];
        yy = [st st fn fn st];
        plot(xx,yy,'k-','linewidth',2)
        st = fn+0.5;
    end
    
    % finish
    s = sprintf('transition probability\nP(bt)-P(wn)), p=%.3g\nmu=%.3g+%.3g',p,mu,p);
    title(s)
    xlabel('P(S@t)')
    ylabel('P(S@t+1 | S@t)')
    
    axis square
    set(gca,'xtick',1:nstate,'xticklabel',leafOrder)
    set(gca,'ytick',1:nstate,'yticklabel',leafOrder)
    
    set_bigfig(gcf,[0.5 0.7])
    
    if saveFig
        sname = [figdir '/transition_prob'];
        save2pdf(sname,gcf)
    end
end

%% correlation of transition probability and embedding distance
if plotCorrTranProbDistance
    % get probabilities
    po   = calculate_cond_trans_prob(C_state, [],nstate,1);
    
    % distance
    D2 = squareform(D);
    
    % correlate
    x = po(:);
    y = D2(:);
    
    corrtype = 'spearman';
    [r,p] = corr(x,y,'type',corrtype);
    
    % plot
    figure
    scatterhist(x,y)
    
    s = sprintf('%s corr, mediod distance vs P(transition)\nR=%.3g, p=%.3g',corrtype,r,p);
    title(s)
    xlabel('trans prob')
    ylabel('distance')
    
    if saveFig
        sname = [figdir '/corr_tranProb_dist'];
        save2pdf(sname,gcf)
    end
end

%% correlate within/between tran prob with dendrogram cutoff
%{
if plotModularity
    nrand = 100;
    
    theseHeights = unique(ceil(tree(:,3)));
    theseHeights(end) = [];
    
    step = 0.5;
    nclust = step:step:floor(max(tree(:,3)));
    %nclust = 8:-1:1;
    %nclust = 2:nstate;
    nsmp = numel(C_state);
                
    % transition
    po = calculate_cond_trans_prob(C_state, [],nstate,1);
        
    % init
    Q = [];
    Qr = [];
    for ih=1:numel(nclust)
        %tClustTmp = cluster(tree,'maxclust',nclust(ih)); 
        tClustTmp = cluster(tree,'Cutoff',nclust(ih),'Criterion','distance'); 

        % modularity
        Q(ih,1) = modularity(po, tClustTmp);

        % rand
        for ir=1:nrand
            idx = randperm(numel(C_state));
            tmp = C_state(idx);

            ptmpr = calculate_cond_trans_prob(tmp, [],nstate,1);
            Qr(ih,ir) = modularity(ptmpr, tClustTmp);
        end
    end
    % stats
    %[mu,se] = avganderror(Q,'mean',2);
    mu = Q;
    
    pval = sum(Qr>Q,2) ./ nrand;
    
    mup = mu;
    mup(pval>0.05) = nan;
    
    % plot
    figure; 
    %shadedErrorBar(nclust,mu,se); 
    plot(nclust,mu,'k.-','linewidth',2,'markersize',15)
    hold all
    plot(nclust,mup,'ro','markersize',10,'linewidth',2)
    plotcueline('y',0)
    
    xlabel('dendrogram cutoff')
    ylabel('modularity')
    
    if saveFig
        sname = [figdir '/tranProb_modularity'];
        save2pdf(sname,gcf)
    end
end
%}


%% hierarchihcal clustering on transitions
if plotModularityHierarchy
    mpath = [anadir '/modularity'];
    if ~exist(mpath); mkdir(mpath); end
    
    if calcModularityHierarchy
        nrand = 50;
        %lags = [1:1:49, 50:2:100, 110:10:1000];
        lags = [1:1:49, 50:2:100, 110:10:200, 300:100:1000];
        
        % calculate over multiple lags
        fprintf('getting all transitions...')
        nsmp = numel(lags)*max(idat);
        PO = nan(nstate,nstate,nsmp);
        ii = 0;
        for id=1:max(idat)
            for ilag = 1:numel(lags)
                sel = idat(istate)==id;
                c = C_state(sel);
                thisLag = lags(ilag);
                
                po = calculate_cond_trans_prob(c, [],nstate,thisLag);
                
                ii = ii+1;
                PO(:,:,ii) = po;
            end
        end
        
        % call python
        verbose = 1;
        cmd = 'fit_louvain';
        dataname = [mpath '/po_obs'];
        savepath = mpath;

        % send to python
        name_in = [dataname '_in.mat'];
        name_out = [dataname '_out.mat'];

        dat = [];
        dat.po = PO;
        save(name_in,'-struct','dat')

        func = [get_code_path() '/bhv_cluster/matlab/python/call_transition_cluster.py'];
        commandStr = sprintf('%s %s %s %s %s',pyenvpath,func,cmd,dataname,savepath);

        tic
        if verbose
            [status,result] = system(commandStr,'-echo');
        else
            [status,result] = system(commandStr);
        end

        % wass it succesful?
        if status~=0 % fail
            error('FAIL')
        end
        toc
        
        % load back
        data_mod = load(name_out);

        % reformat data
        m = data_mod.modularity;
        
        m = reshape(m,numel(lags),max(idat)); % lags X data
        [mu,se] = avganderror(m,'mean',2);
        
        figure
        shadedErrorBar(lags,mu,se)
        set(gca,'xscale','log')
        
        foo=1;
    end
end

%% modularity vs lag
if plotModularityLag
    
    sname = [anadir '/modularity.mat'];

    if calcModularity
        nrand = 100;
        %lags = [1:1:49, 50:2:100, 110:10:1000];
        lags = [1:1:49, 50:2:100, 110:10:200, 300:100:1000];
        

        nclust = 2:nstate;

        % init        
        Q = nan(numel(nclust),numel(lags),max(idat));
        Qr = nan(numel(nclust),numel(lags),max(idat),nrand);
        Qv = cell(numel(nclust),numel(lags),max(idat));
        Qvr = cell(numel(nclust),numel(lags),max(idat),nrand);

        % observed
        fprintf('modularity at all lags')
        for id=1:max(idat)
            dotdotdot(id,0.1,max(idat))

            sel = idat(istate)==id;
            c = C_state(sel);

            for ilag=1:numel(lags)  
                thisLag = lags(ilag);
                
                % transition
                po = calculate_cond_trans_prob(c, [],nstate,thisLag);

                for ih=1:numel(nclust)
                    tClustTmp = cluster(tree,'maxclust',nclust(ih)); 
                    %tClustTmp = cluster(tree,'Cutoff',nclust(ih),'Criterion','distance'); 

                    % modularity
                    [q,qv] = modularity(po, tClustTmp);
                    Q(ih,ilag,id) = q;
                    Qv{ih,ilag,id} = qv;
                    
                    % rand
                    parfor ir=1:nrand
                        idx = randperm(numel(c));
                        tmp = c(idx);

                        ptmpr = calculate_cond_trans_prob(tmp, [],nstate,thisLag);
                        [q,qv] = modularity(ptmpr, tClustTmp);
                        Qr(ih,ilag,id,ir) = q;
                        Qvr{ih,ilag,id,ir} = qv;
                        
                    end
                end
            end
        end

        % save
        save(sname,'Q','Qr','Qv','Qvr','lags')
    else
        load(sname)
    end

    % stats
    [Qmu,Qse] = avganderror(Q,'mean',3); 
    [Qrmu,Qrse] = avganderror(Qr,'mean',3); 
        
    p = sum(Qrmu >= Qmu,4) ./ size(Qrmu,4);
    p = bonf_holm(p);
    
    % ---------------------------------------------------------
    % plot all modualrities
    %{
    cmap = colormap('copper');
    cmap = cmap(ceil(linspace(1,size(cmap,1),numel(lags))),:);
    for ip=1:numel(lags)
        plot(nclust,Qmu(:,ip),'color',cmap(ip,:))
        hold all
    end
    %}
    
    figure
    nr = 1; nc = 1;
    set_bigfig(gcf,[0.4 0.5])
    
    subplot(nr,nc,1)
    imagesc(lags,nclust,Qmu,'alphadata',p<0.05)
    
    colorbar
    axis square
    
    title('modularity as a func of lag and partition size')
    xlabel('lag')
    ylabel('partition size')
    set(gca','ytick',[nclust(1):2:nclust(end)])
    %set(gca,'colorscale','log')
    
    if saveFig
        sname = [figdir '/tranProb_modularity_manyLags'];
        save2pdf(sname,gcf)
    end
   
    
    % ---------------------------------------------------------
    % find significant modules
    iclust = 1;
    ilag = 1;
    
    qv = Qv(iclust,ilag,:);
    qv = cat(2,qv{:});
    
    qvr = Qvr(iclust,ilag,:,1);
    qvr = cat(2,qvr{:});
    qvr = reshape
    

end


%% probability of revisiting state after T transitions
if plotProbRevisitState
    disp('plotting markov...')
    
    % settings
    thisEig = 1; % for decay stats
    icluster = 1; % for plotting
    
    neig = 6;
    %taus = 1:2:1000;
    taus = [2:100, 100:10:1000];
    %taus = 1:10:101;
    
    nclust = nstate; %nstate:-4:3;

    nlab = nclust(1);
    udat = unique(idat(~isnan(idat)));
    idat_state = idat(istate);
    
    % get transition matrices
    if doCalculations
        % get cladogram        
        Pobs = {};
        Pexp = {};

        for ic=1:numel(nclust)
            dotdotdot(ic,0.1,numel(nclust))

            % cluster based on poses and remap
            t = cluster(tree,'MaxClust',nclust(ic));
            theseStates = C_state;
            theseStates = changem(theseStates,t,1:nstate);


            % get expected/ markov-predicted probabilities
            for id=1:numel(udat)
                sel = idat_state==udat(id);
                tmp = theseStates(sel);
                tmp(isnan(tmp)) = [];

                po_orig = calculate_cond_trans_prob(tmp, [],nclust(1),1);

                for it=1:numel(taus)
                    % probs matrix size will be as large as the most
                    % clusters
                    t = taus(it);
                    po = calculate_cond_trans_prob(tmp, [],nclust(1),t);
                    pe = po_orig^t;

                    % store
                    Pobs{id,it,ic} = po;
                    Pexp{id,it,ic} = pe;
                end
            end
        end
    end
    
    % get mean within each dataset, and normalize
    %{
    MUobs = [];
    MUexp = [];
    for id=1:size(Pobs,1)
        for it=1:size(Pobs,2)
            % data
            tmp1 = Pobs{id,it};
            tmp2 = Pexp{id,it};
            
            % statioary probabilities
            mu = nanmean(Pexp{id,end},2);
            
            % normalize
            tmp1 = tmp1 ./ mu;
            tmp2 = tmp2 ./ mu;
            
            foo=1;
        end
    end
    %}
    
    % get mean within each dataset
    MUobs = cellfun(@(x) nanmean(diag(x)),Pobs);
    MUexp = cellfun(@(x) nanmean(diag(x)),Pexp);
    
    % normalize so that it decays to zero at infinity
    
    MU = {MUobs,MUexp};
    
    % plot mean over datasets
    cols = get_safe_colors(0,1:2);
    mk = {'-',':'};
    
    figure
    h = [];
    for ii=1:2
        tmp = MU{ii};
        [mu,se] = avganderror(tmp,'mean');
        
        htmp = shadedErrorBar(taus,mu,se,{'color',cols(ii,:)},1);
        h(ii) = htmp.mainLine;
        hold all
    end
    
    axis square
    legend(h,{'obs','pred'})
    set(gca,'xscale','log')
    
    title('probability of returning to the same state')
    xlabel('# transitions')
    ylabel('mean probability of return')
    set(gca,'fontsize',14)
    
    % save
    if saveFig
        sname = sprintf('%s/markov_prob_return_same_state',figdir);
        save2pdf(sname,gcf)
    end

end


%% detect sequences
if plotSequenceProb
    tmp = C_state;
    po = calculate_cond_trans_prob(tmp, [],nstate,1);
    G = digraph(po);
    
    [S, C] = graphconncomp(sparse(po));
    
end





%% markovian analysis
if plotMarkovTimescales
    disp('plotting markov...')
    
    % settings
    thisEig = 1; % for decay stats
    icluster = 1; % for plotting
    
    neig = 6;
    %taus = 1:2:1000;
    taus = [1:100, 100:10:1000];
    %taus = 1:10:101;
    
    nclust = nstate; %nstate:-4:3;

    nlab = nclust(1);
    udat = unique(idat(~isnan(idat)));
    idat_state = idat(istate);
    
    % get transition matrices
    if doCalculationsMarkov
        % get cladogram        
        Pobs = {};
        Pexp = {};

        for ic=1:numel(nclust)
            dotdotdot(ic,0.1,numel(nclust))

            % cluster based on poses and remap
            t = cluster(tree,'MaxClust',nclust(ic));
            theseStates = C_state;
            theseStates = changem(theseStates,t,1:nstate);


            % get expected/ markov-predicted probabilities
            for id=1:numel(udat)
                sel = idat_state==udat(id);
                tmp = theseStates(sel);
                tmp(isnan(tmp)) = [];

                for it=1:numel(taus)
                    % probs matrix size will be as large as the most
                    % clusters
                    t = taus(it);
                    po   = calculate_cond_trans_prob(tmp, [],nclust(1),t);
                    po(isnan(po)) = 0; % never visited
                    pe = po^t;

                    % store
                    Pobs{id,it,ic} = po;
                    Pexp{id,it,ic} = pe;
                end
            end

        end
        
        % get eigenvalues of each
        E1 = cellfun(@(x) sort(abs(eig(x)),'descend'), Pobs,'un',0);
        E1 = reshape(cat(2,E1{:}),[nlab,numel(udat),numel(taus),numel(nclust)]);

        E2 = cellfun(@(x) sort(abs(eig(x)),'descend'), Pexp,'un',0);
        E2 = reshape(cat(2,E2{:}),[nlab,numel(udat),numel(taus),numel(nclust)]);

        E = {E1,E2};
    
         % ---------------------------------------------------------
        % get the half life
        %thisEig = 5+1;
        doFitExp = 0;

        lim = [0 50];
        thresh = 0.5;
        P50 = [];
        for thisEig=1:size(E1,1)-1
            dotdotdot(thisEig,0.1,size(E1,1)-1)
            for jj=1:2
                for id=1:numel(udat)
                    for ic=1:numel(nclust)

                        if doFitExp % fit exponential
                            sel = taus >= lim(1) & taus <= lim(2);
                            tmp = E{jj}(thisEig+1,id,sel,icluster);
                            tmp = squeeze(tmp);

                            if all(tmp==0)
                                p50 = Inf;
                            else
                                f = fit(taus(sel)',tmp(sel),'exp1');
                                p50 = -log(2)./f.b;
                            end
                            %figure; plot(taus(sel),tmp(sel)); hold all; plot(taus(sel),f(taus(sel)))

                        else % 50% point
                            tmp = E{jj}(thisEig+1,id,:,ic);
                            tmp = squeeze(tmp);

                            mx = tmp(1)*thresh;
                            p50 = find(tmp < mx,1);

                            if isempty(p50)
                                p50 = nan;
                            else
                                p50 = taus(p50);
                            end
                        end

                        % store
                        P50(thisEig,id,jj,ic) = p50;
                    end
                end
            end
        end
    end

    % ---------------------------------------------------------
    % plot example transition matrices
    po   = calculate_cond_trans_prob(C_state, [],nstate,1);
    po2 = po(:,leafOrder);
    po2 = po2(leafOrder,:);
    
    figure
    nr = 2; nc = 2;
    
    tt = [1 100 1000];
    strs = {'Markov Prediction for tau=100','Observed prob, tau=1','Observed probability for tau=100','Observed probability for tau=1000'};
    for ii=1:4
        if ii==1
            p = po2^100;
        else
            p = calculate_cond_trans_prob(C_state, [],nstate,tt(ii-1));
            p = p(:,leafOrder);
            p = p(leafOrder,:);
        end
            
        % plot
        ns = ii;
        subplot(nr,nc,ns)
        
        xx = 1:nstate;
        imagesc(xx,xx,p)
        set(gca,'xtick',xx,'xticklabel',leafOrder)
        set(gca,'ytick',xx,'yticklabel',leafOrder)
        
        axis square
        
        title(strs{ii})
        xlabel('Final state')
        ylabel('Initial state')
        colorbar
    end
    setaxesparameter('clim')
    set_bigfig(gcf)

    if saveFig
        sname = sprintf('%s/markov_obsVSpred_transMat',figdir);
        save2pdf(sname,gcf)
    end

    
    % ---------------------------------------------------------
    % plot eigenvalues as a function of transition
    figure
    doTrans = 1;
    cols = get_safe_colors(0,1:neig);
    mk = {'-','--'};
    h = [];
    for ii=1:neig
        for jj=1:2
            tmp = squeeze( E{jj}(ii+1,:,:,icluster) );
            [mu,se] = avganderror(tmp,'mean');
            %[mu,se] = avganderror(tmp,'median',1,1,200);
            
            htmp=shadedErrorBar(taus,mu,se,{mk{jj},'linewidth',2,'color',cols(ii,:)},doTrans);
            h(ii,jj) = htmp.mainLine;
            hold all
        end     
    end
    
    s = cellfun(@(x) ['\lambda_' num2str(x+1)],num2cell(1:neig),'un',0);
    legend(h(:,1),s)
    set(gca,'xscale','log')
    
    xlabel('# transitions')
    ylabel('|\lambda|')
    title('eigenvalue for observed vs Markov expectation')
    set(gca,'fontsize',14)
    
    if saveFig
        sname = [figdir '/markov_decay'];
        save2pdf(sname,gcf)
        
    end
    
    % ---------------------------------------------------------
    % EXAMPLE: plot eigenvalues as a function of transition
    if 0
        figure
        p50 = [];
        c = 'k';
        for jj=2:-1:1
            doTrans = 0;
            cols = get_safe_colors(0,1:neig);
            mk = {'-','--'};

            tmp = squeeze( E{jj}(2,:,:,icluster) );
            [mu,se] = avganderror(tmp,'mean');
            %[mu,se] = avganderror(tmp,'median',1,1,200);
            p50(jj) = taus(find(mu < mu(1)*0.5,1));

            %htmp=shadedErrorBar(taus,mu,se,{mk{jj},'linewidth',2,'color',cols(ii,:)},doTrans);
            plot(taus,mu,mk{jj},'linewidth',2,'color',c)
            h(ii,jj) = htmp.mainLine;
            hold all

            set(gca,'xscale','log')

            xlabel('# transitions')
            ylabel('|\lambda|')
            title('eigenvalue for observed vs Markov expectation')
            set(gca,'fontsize',14)

            % save figure
            sname = sprintf('%s/example_eig_decay_%g',figdir,jj);
            save2pdf(sname)

            if jj==1
                plotcueline('x',p50(1),mk{1},'color',c)
                plotcueline('x',p50(2),mk{2},'color',c)
                
                sname = sprintf('%s/example_eig_decay_%g',figdir,0);
            	save2pdf(sname)
            end
        end
    end
    
   
   
    % ------------------------------------------------------------
    % plot average over all, for slow timescale
    p50 = squeeze(P50(1,:,:,1));
    [mu,se] = avganderror(p50,'mean');
    
    [~,p] = ttest(p50(:,1),p50(:,2));
    
    figure
    barwitherr(se,1:2,mu)
    set(gca,'xlim',[0.5 2.5],'xtick',1:2,'xticklabel',{'obs','markovPred'})
    axis square
    
    s = sprintf('# transitions to reach 50%% of max\nmu+se=%s+%s\np=%.3g',...
        mat2str(mu,3),mat2str(se,3),p);
    title(s)
    ylabel('p50 (transitions)')
    
    % task differences
    d = p50(:,1) - p50(:,2);
    m = taskInfo.ck;
    t = taskInfo.Condition;
    
    t2=t; 
    if 1
        theseTasks = {{'no_task'},{'feeder'},{'rand_feeders'}};
        theseTasksStr = {'no task','feeder','rand_feeders'};
    else
        t2(strcmp(t,'rand_feeders')) = {'feeder'};
        theseTasks = {{'no_task'},{'feeder','rand_feeders'}};
        theseTasksStr = {'no task','task'};
    end
    
    g = {m,t2};
    [P,T,STATS,TERMS] = anovan(d,g,'model','interaction');
    


    MU=[]; SE=[]; DAT = {};
    for it=1:numel(theseTasks)
        sel = ismember(t,theseTasks{it});
        tmp = d(sel);
        DAT{it} = tmp;
        [MU(it),SE(it)] = avganderror(tmp,'mean');
    end
    
    figure
    barwitherr(SE,1:numel(theseTasks),MU)
    set(gca,'xticklabel',theseTasksStr)
    
    % ------------------------------------------------------------
    % plot difference as a function of eigenvalue
    d = P50(:,:,1) - P50(:,:,2);
    P = [];
    for ii=1:10%size(P50,1)
        tmp = d(ii,:);
        [~,T,~,~] = anovan(tmp,g,'model','interaction','display','off');
        P(ii,1) = T{2,7};
        P(ii,2) = T{3,7};
        P(ii,3) = T{4,7};
    end
    legstr = {'monk','task','monk*task'};
    
    figure
    plot(P,'.-')
    plotcueline('y',0.05)
    legend(legstr)
    
    % ------------------------------------------------------------
    % plot decay diff as a function of cluster number
  
    
    % plot
    doTrans = 1;
    
    figure
    cols = get_safe_colors(0,[1 2 5]);
    cols = [zeros(1,3); cols];
    legstr = [{'all'},theseTasksStr];
    h = [];
    for it=1%:numel(theseTasks)+1
        if it>1
            sel = ismember(t2,theseTasks{it-1});
        else
            sel = true(size(t2));
        end
            
        % info
        d = squeeze(P50(1,sel,1,:) - P50(1,sel,2,:));
        [MU,SE] = avganderror(d,'mean');
        [~,P] = ttest(d);

        D = d(:);
        N = repmat(nclust(1:end-1),[size(d,1) 1]);
        N = N(:);

        mx = max(MU+SE);
        mup = ones(size(MU))*mx*1.05;
        mup(P>0.05) = nan;

        % plot
        htmp = shadedErrorBar(nclust,MU,SE,{'.-','color',cols(it,:),'markersize',10},doTrans);
        h(it) = htmp.mainLine;
        hold all
        plot(nclust,mup,'*','color',cols(it,:))
    end
    
    legend(h,legstr)
    
    s = sprintf('excess decay vs nCluster\nPearson R=%.3g, p=%.3g',R,pc);
    title(s)
    xlabel('nClusters')
    ylabel('excessive decay')
    
    % save
    if saveFig
        sname = sprintf('%s/nclust_vs_excessiveDecay',figdir);
        save2pdf(sname,gcf)
    end
end



    
    
%% embed sequence
if plotEmbedSeq
    sparsePlot = 0;
    
    %{
    - for ultiple sequence lengths, ask: can we predict the next sequences
        - at some point, it should not be preditble anymore
    %}
    
    %seqFeat = 'state-dur'; %state, state-onehot, state-dur
    
    % load back info to plot  
    fprintf('\tloading...\n')

    inx = load([seqdir '/X.mat']);
    iny = load([seqdir '/umap_train.mat']);
    inc = load([seqdir '/cluster_train.mat']);
    load([seqdir '/info.mat'])
    
    A2 = inx.X;
    Y_state = iny.embedding_;
    
    clabels_seq = inc.labels;
    
    nlabel_seq = accumarray(clabels_seq,1);
    p = nlabel_seq ./ sum(nlabel_seq);
    bad = p < 0.01;
    clabels_seq(bad) = nan;
    
    oldval = 1:max(clabels_seq);
    newval = nan(size(oldval));
    newval(~bad) = 1:sum(~bad);
    clabels_seq = changem(clabels_seq,newval,oldval);
    
    Ld_seq = changem(inc.Ld,newval,oldval);
    %Lbnds_seq = changem(inc.Lbnds,newval,oldval);
    
    nseq = max(clabels_seq);
    
    % convert back to states
    if strcmp(seqFeat,'state-onehot')
        A_state2 = nan(size(A2,1),nwin);
        for ii=1:nwin
            st = (ii-1)*nstate + 1;
            fn = (ii)*nstate;

            tmp = A2(:,st:fn);
            [~,idx] = max(tmp,[],2);
            A_state2(:,ii) = idx;
            foo=1;
        end
    elseif strcmp(seqFeat,'state-dur') || strcmp(seqFeat,'state')
        A_state2 = A2;
    end
    
    % plot embedding
    fprintf('\tplotting...\n')

    if plotSeqEmbedding
        figure('name','seq embeddding')
        plot_embedding(inc.outClust,inc.Lbnds,Ld_seq);
    end
    
    % monkey specificity
    if plotSeqMonkeySpecificity
         % monkey specificity
        figure('name','seq-residency')
        set_bigfig(gcf,[0.7 0.5])
        nr = 2; nc = 1;

        nclust = accumarray(clabels_seq(~isnan(clabels_seq)),1);
        [n,xe,ye] = histcounts2(clabels_seq,idat_seq(:,1),[max(clabels_seq) max(idat_seq(:,1))]);
        p = n ./ nclust;

        p2 = nclust ./ sum(nclust(:));

        %p = p ./ accumarray(idat,1)';
        th = 1/max(idat);
        b = mean(abs(p-th),2) * 100;
        b = sum(p2.*b);
        
        cols = get_safe_colors(0,[1 2]);
        monks = {'yo','ca'};
        monk = cellfun(@(x) x(1:2), {datasets.name},'un',0);

        subplot(nr,nc,1)
        for id=1:max(idat)
            im = ismember(monks,monk(id));
            tmpp = p(:,id);
            plot(1:max(clabels_seq),tmpp,'.-','markersize',10,'color',cols(im,:))
            hold all
        end

        plotcueline('y',th,'k:')
        str = sprintf('probability of residency by monk\nweighted bias=%.3g %%',b);
        title(str)

        legstr = cellfun(@(x) x(1:2), {datasets.name},'un',0);
        legend(legstr,'location','eastoutside')
        pause(0.1)
        pos1 = get(gca,'position');
    
        % residency
        subplot(nr,nc,2)

        plot(p2,'.-','markersize',10)
        title('residency')
        pos2 = get(gca,'position');
        pos2 = [pos2(1:2) pos1(3) pos2(4)];
        set(gca,'position',pos2)

    end
    
    % plot sequence
    if plotSeqs
        figure('name','seqs')
        nr = nwin; nc = max(clabels_seq);
        %tiledlayout(nr,nc, 'Padding', 'none', 'TileSpacing', 'compact');

        for ic=1:max(clabels_seq)
            sel = inc.labels==ic;
            tmp = A_state2(sel,1:nwin);

            % most common seq
            [useq,~,iseq] = unique(tmp,'rows');
            nseq = accumarray(iseq,1);
            [~,imx] = max(nseq);
            mu = useq(imx,:);
            
            b = nseq(imx) ./ sum(nseq);
            
            for iw=1:nwin
                idx = mu(iw);
                p = pose_mu(idx,:);

                ns = ic + (iw-1)*nc;
                subplot(nr,nc,ns)
                %nexttile(ns)
                plot_monkey({p,labels},[],1);
                set(gca,'xtick',[],'ytick',[],'ztick',[])

                if ~sparsePlot
                    s = sprintf('seq%g, %g\nbias=%.3g',ic,idx,b);
                    title(s)
                end
            end
            foo=1;
        end
        set_bigfig(gcf,[1 0.7])
    end
    
    % is it hierarchihcal? some kind of cross-over metric
    if plotSeqTree
        [~,~,ic] = unique([A_state2(:)]);
        nstate_tot = accumarray(A_state2(:),1,[nstate, 1]);
        
        nstate_seq = [];
        for iseq=1:nseq
            sel = clabels_seq==iseq;
            a = A_state2(sel,:);
            tmp = accumarray(a(:),1,[nstate 1]);
            nstate_seq(:,iseq) = tmp;
        end
        pstate_seq = nstate_seq ./ nstate_tot;
        
        %pstate_seq = pstate_seq(leafOrder,:);
        
        % plot tree
        figure('name','seq tree')

        
        cmap = colormap();
        %idx = ceil(linspace(1,size(cmap,1),nstate));
        idx = tClust .* floor(size(cmap,1)./max(tClust));
        cmap = cmap(idx,:);
        
        xseq = 1:3:nseq*3;

        plot(xseq,ones(size(xseq))*2,'k.','markersize',20)
        hold all
        
%         for iseq=1:nseq
%             plot(xseq(iseq),2,'.','color',cmap(iseq,:),'markersize',20)
%             hold all
%         end
        
        % plot child level
        xstate = 1:nstate;
        %xstate = leafOrder;
        plot(xstate,ones(size(xstate)),'k.','markersize',20)

        for iseq=1:nseq
            for istate=1:nstate
                p = pstate_seq(leafOrder(istate),iseq);
                
                w = ceil(p./0.25);
                c = cmap(leafOrder(istate),:);
                
                %x = [xseq(iseq), xstate(istate)];
                x = [xseq(iseq), xstate(istate)];
                y = [2 1];
                if w > 0
                    plot(x,y,'linewidth',w,'color',c)
                end
            end
        end
        
    end
    
end


