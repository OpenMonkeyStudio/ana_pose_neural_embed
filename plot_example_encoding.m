function plot_example_encoding(datasets,thisData,figdir,SDF,C_in,idat)

% stuff
id = find(contains({datasets.name},thisData));
thisTest = id;
areaOrder = {'PM','SMA','DLPFC','VLPFC','ACC','OFC'};

monk = thisData(1:2);

% flags

loadStuff = 0;
    loadConcat = 0;

plotExampleRate = 1;

%% testing lims

%% define what we'll be testing
if strcmp(monk,'yo')
    if thisTest==8 %1.5min, walk, jump up/dwn, sit, feeder 

        if 1 % test=1
            plottingLims = [13179,13205;13220,13535;13535,14335;14435,14464;14469,15465]';
            plottingStrs = {'jump down','walk','feeder','jump up','sit'};
        end
        
    elseif thisTest==10 % yoda 2/28/2022
        plottingLims = [1 1000; 1110 1126; 1150 1465; 1530 2617; 2640 2710; 2722 2740 ]';
        plottingStrs = {'sit','jump down','walk','feeder','walk','jump up'};
    elseif thisTest==78
        
        plottingLims = [0 1100; 1104 1123; 1125 1178; 1337 2050; 2081 2280; 2285 2310]';
        plottingStrs = {'sit','jump down','walk','feeder','walk','jump up'};
    else
        error('oops')
    end
elseif strcmp(monk,'wo')
    if thisTest==21 % wo 12/8/2021
        plottingLims = [1388 1580; 1680 1705; 1740 1900; 7600 7880; 8645 8679; 10193 11000; 11008 11028; 34530 34620]';
        plottingStrs = {'walk','jump up','stand4','feeder','jump down (head)','hang wall','jump down','climb'};
    elseif thisTest==22
        plottingLims = [1680 1705; 8645 8679; 10080 10109; 11008 11028; 11700 11739; 13665 13700]'; 
        plottingStrs = {'jump up', 'jump down (head down)', 'jump up', 'jump down', 'jump up', 'jump down'};
    elseif thisTest==23
        plottingLims = [1388 1580; 8700 8800; 8900 9040; 11050 11200]'; 
        plottingStrs = {'walk','walk','walk','walk'};
    end
else
    error('oops')
end

[~,is] = sort(plottingLims(1,:));
plottingLims = plottingLims(:,is);
plottingStrs = plottingStrs(is);


%% plot mean rate
if plotExampleRate    
    % select
    sel = find(contains({SDF.name},thisData));
    sdf = SDF(sel);
    f = sdf(1).frame;
    id = sdf(1).id;
    
    a = [sdf.area];
    a = collapse_areas(a);
    
    % mean rate for each lim
    X = cat(1,sdf(:).sdf);
    
    mu = [];
    for ip=1:size(plottingLims,2)
        lim = plottingLims(:,ip);
        selt = f >= lim(1) & f <= lim(2);
        
        m = nanmean(X(:,selt),2);
        mu(:,ip) = m;
    end
    
    % anova on each cell
    c = C_in(idat==id);
    
    ff = [];
    for ii=1:size(X,1)
        Xt = X(ii,:);
        [~,T,~] = kruskalwallis(Xt,c,'off');
        ff(ii) = T{2,5};
    end
    
    % prep
    mu = mu ./ sum(abs(mu),2);
    
    [~,ia] = ismember(a,areaOrder);
    %[~,is] = sort(ia);
    [~,is] = sortrows([ia;ff]');
    
    mu = mu(is,:);
    a = a(is);
    ia = ia(is);
    ff = ff(is);
    
    idiff = find(diff(ia)~=0);
    idiff2 = [1 idiff numel(ia)];
    idiff2 = ceil( nanmean([idiff2(1:end-1); idiff2(2:end)]) );
    ua = a([1 idiff+1]);

    % plot
    figure
    nr = 1; nc = 2;
    set_bigfig(gcf,[0.6 0.4],[0 0.3])
    cols = get_safe_colors(0,[1:5 7]);

    % plot mean rate per example actions
    subplot(nr,nc,1)
    imagesc(mu)
    set(gca,'xtick',1:size(mu,2),'xticklabel',plottingStrs)
    set(gca,'ytick',idiff2,'yticklabel',areaOrder)
    xtickangle(45)
    pcl('y',idiff+0.5)
    axis square
    colorbar
    
    n = accumarray(ia',1)';
    s = sprintf('mean resid rate for example actions\nN=%g,n=%s',size(mu,1),mat2str(n));
    title(s)
    
    % plot mean per area
    subplot(nr,nc,2)
    %plot(ff)
    %pcl('x',idiff+0.5)
    hv = violinplot(ff,ia);
    for ii=1:numel(hv)
        hv(ii).ViolinPlot.FaceColor = cols(ii,:);
        set(hv(ii).ScatterPlot,'MarkerFaceColor',cols(ii,:))
        %hv(ii).ScatterPlot.Marker = cols(ii,:);
    end
    set(gca,'xtick',1:numel(ua),'xticklabel',ua)
    axis square
    
    title('action encoding, KW test')
    ylabel('mean F')
    
    % save
    sname = [figdir '/' thisData '_example_encode.pdf'];
    save2pdf(sname)
    
    foo=1;
end