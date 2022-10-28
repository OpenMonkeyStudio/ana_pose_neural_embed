function plot_embedding_yo_02_25_2021(anadir)

%% settings
datadir = [fileparts(anadir) '/Data_proc_13joint'];

% flags
loadStuff = 1;

thisID = 8; %yo: 8(3,5), 11(11) | wo: 3(21)
thisTest = 1;

plotClusterLabel = [0 1];

% plooting flags
saveFig = 1;

plotEmbedding = 1;
saveFrames = 1;
plotFastResidency = 0;
plotMeanVelocity = 0;

%% load data
if loadStuff
fprintf('loading...\n')
    clustInfo = load([anadir '/cluster_train.mat']);
    C_in = clustInfo.clabels;
    umap_train = load([anadir '/umap_train.mat']);
    Y_train = umap_train.embedding_;
    
    info = load([anadir '/info.mat']);
    try,info = rmfield(info,'info'); end
    evals(info)
    
    % original data
    name = datasets(thisID).name;
    sname = [datadir '/data_ground/' name '_proc.mat'];
    load(sname)
    labels = data_proc.labels;
    
    % features
    sname = [anadir '/X_feat_norm/' name '_feat.mat'];
    in1 = load(sname);
    
    % videos
    vidnames = {'vid_18261112_full.mp4','vid_18261030_full.mp4'};
    vidnames = cellfun(@(x) [fileparts(anadir) '/' name '/vids/' x],vidnames,'un',0);    
end

idat_train = info.idat(info.idx_train);
frame_train = info.frame(info.idx_train);

sel1 = info.idat==thisID;
sel2 = idat_train==thisID;

f1 = info.frame(sel1);
f2 = frame_train(sel2);
C = C_in(sel2);
Y = Y_train(sel2,:);
x = in1.X_feat(ismember(data_proc.frame,f2),:);

%% define what we'll be testing
if thisTest==1 %1.5min, walk, jump up/dwn, sit, feeder 
    
    %testLim = [13050 15500]; % perfet
    %testLim = [13050 19000];
    %jumpLim = [13151 13215; 14440 14470];
    plottingLims = [13179,13205;13220,13535;13535,14335;14390,14430;14435,14464;14469,14915;14965,15465]';
    plottingStrs = {'jump down','walk','feeder','walk','jump up','sit1'};

    testLim = [min(plottingLims(:)) max(plottingLims(:))];
elseif thisTest==3 %confusing limbing with standing upright/sitting plsayed open

    testLim = [0 7000];
    jumpLim = [];
    plottingLims = [1000 1100; 1140 1160; 2200 2400; 2600 2620; 2630 2680; 5530 5600; 5650 5900; 6230 6320]';
    plottingStrs = {'stand upright', 'jump up', 'sit w foot','jump down','walk','climb right', 'hangWall', 'climb left'};
    %2400 2490; 'stand up'

elseif thisTest==5
    testLim = [2500 14500];
    
    plottingLims = [2630 2680; 13220 13535; 14390 14430; 7240 7350;7405 7590; 5650 5900; 6230 6320]';
    plottingStrs = {'walk','walk','walk','climb left', 'climb right','climb right','climb left'};
end

[~,is] = sort(plottingLims(1,:));
plottingLims = plottingLims(:,is);
plottingStrs = plottingStrs(is);

lim = testLim;




%% sample embedding points
if plotEmbedding
    fprintf('plotting embedding overlaid with actions...\n')

    figure
    nr = 2; nc = 2;
    set_bigfig(gcf,[0.35 0.4],[0 0.25]);

    % embedding with points
    subplot(nr,nc,1)
    out = plot_embedding(clustInfo.outClust,clustInfo.Lbnds,clustInfo.Ld,plotClusterLabel,1);
    set(out.himg,'alphadata',ones(size(clustInfo.outClust))*0.2)

    %axis(out.hax,'off')
    colorbar(out.hax,'off')
    newpos = [0.1 0.5 0.45 0.45];
    set(out.hax,'position',newpos)
    set(out.htext,'fontsize',10)
    hold all

    idx = {}; pl = plottingLims; 
    for ii=1:size(pl,2)
        idx{ii} = nearest(f2,pl(1,ii)):nearest(f2,pl(2,ii));
    end

    jj = cellfun(@numel,idx(1:end)); jj=cumsum(jj);
    tt = floor(mean([0 jj(1:end-1);jj]));
    %str = cellfun(@(x) sprintf('%g-%g',x(1),x(end)),idx,'un',0);
    str = plottingStrs;
    legstr = cellfun(@(x,y) sprintf('%s: %g-%g',x,y(1),y(end)),plottingStrs,idx,'un',0);

    ff = cellfun(@(x) x(1:strfindk(1,x,'_')-1),feat_labels,'un',0);
    [ufeat,~,ifeat] = unique(ff);
    ifeat2 = [0, find(diff(ifeat)~=0)', numel(ff)-1]+1; ufeat = ff(ifeat2(1:end-1));
    ifeat3 = floor(mean([ifeat2(1:end-1); ifeat2(2:end)]));

    cols = 'rgbcmykw'; h=[];
    for ii=1:numel(idx)
        y = Y(idx{ii},:);
        h(ii)=plot(y(:,1),y(:,2),'.','color',cols(ii),'markersize',10);
    end
    hl = legend(h,legstr,'location','westoutside');
    pos = get(hl,'position');
    pos = [0 0.65 pos(3:4)];
    set(hl,'position',pos);

    if 1
    pos = [0 0.8 0.15 0.15];
    newax = axes();
    out2 = plot_embedding(clustInfo.outClust,clustInfo.Lbnds,clustInfo.Ld,1,1);
    colorbar off
    set(gca,'yticklabel',[],'xticklabel',[])
    axis off
    set(newax,'position',pos)
    end

    % cluster time series
    subplot(nr,nc,2)

    selx = f2 >= lim(1) & f2 <= lim(2);
    CC = C(selx);
    f3 = f2(selx);
    ref = f3(1);
    f3 = f3-ref;

    ylim = [0 max(CC)+1];
    set(gca,'ylim',ylim)
    for ii=1:numel(idx)
        id = idx{ii}; % in f2 index
        [~,id,~] = intersect(f3+ref,f2(id));
        id = f3(id);
        px = [id(1) id(end) id(end) id(1)]; 
        py = [ylim(1) ylim(1) ylim(2) ylim(2)];
        hp=patch(px,py,cols(ii),'facealpha',0.2,'EdgeAlpha',0.5);
        hold all
    end

    plot(f3,CC,'k-','linewidth',1)
    xlabel('frame')
    ylabel('cluster ID')
    set(gca,'xlim',[f3(1) f3(end)])

    % features
    subplot(nr,nc,3)
    tmp=x([idx{:}],:);
    imagesc(tmp); 
    plotcueline('y',jj); plotcueline('x',ifeat2(2:end-1)-0.5)
    axis square; colorbar; set(gca,'ytick',tt,'yticklabel',str,'xtick',ifeat3,'xticklabel',ufeat); xtickangle(90)
    %if 1 && contains(data.normalization,'zscore'); set(gca,'clim',[-5 5]); end
    set(gca,'clim',[-5 5]); 

    % distances
    subplot(nr,nc,4)
    d = pdist(tmp,umap_train.metric); 
    d2 = squareform(d);
    imagesc(d2); 
    plotcueline('y',jj); plotcueline('x',jj)
    axis square; colorbar; set(gca,'ytick',tt,'yticklabel',str,'xtick',tt,'xticklabel',str); xtickangle(90)

    % save
    if saveFig
        sname = sprintf('%s/Figures/example_embed_%s.pdf',anadir,name);
        save2pdf(sname,gcf)
    end
end


%% extract and save frames for extracted behaviours
if saveFrames
    fprintf('saving frames from each action...\n')

    % prep
    fpath = sprintf('%s/Figures/behav_frames',anadir);
    if ~exist(fpath); mkdir(fpath); end
    
    tmp = [];
    tmp.vidnames = vidnames;
    tmp.plottingLims = plottingLims;
    tmp.plottingStrs = plottingStrs;
    tmp.name = name(1:end-9);
    save([fpath '/info.mat'],'-struct','tmp')
    
    % open vids
    V = {};
    for im=1:numel(vidnames)
        V{im} = VideoReader(vidnames{im});
    end

    % loop through defined lims
    tic
    fprintf('extracting frames from video: ')
    for ip=1:size(plottingLims,2)
        fprintf('%g,',ip);
        flim = plottingLims(:,ip);
        fidx = flim(1):5:flim(2);
        
        for im=1:numel(V)
            v = V{im};
            for ii=1:numel(fidx)
                fr = fidx(ii);
                vf = read(v,fr);
                
                % save
                sname = sprintf('%s/cam%g_lim%g_f%g.jpeg',fpath,im,ip,fidx(ii));
                imwrite(vf,sname)
            end
        end

    end
    fprintf('\n')
    toc
end

%% save movie of this whole segment?

% selx = ismember(data_proc.frame,f2);
% selx2 = true(size(C));
% 
% %cstr = {'orient','jump down','sit1','jump up','walk','sit2','sit3','sit4'};
% cstr = cellfun(@num2str,num2cell(1:max(C)),'un',0);
% Fr = data_proc.frame(selx);
% DS = data_proc.data(selx,:);
% C2 = cstr(C(selx2));
% %C2 = {};
% play_movie_pose(vidnames,DS,labels,Fr,C2)



