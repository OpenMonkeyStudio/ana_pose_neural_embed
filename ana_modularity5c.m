
%% flags
saveFig = 1;

calcModularityHierarchy = 0;
    doClustering = 1;
    doHierarchicalClustering = 1;

prepData = 0;
    useCutoffs = 0;
    loadPreppedData = 0;
    trainData = 0;
useHierResults = 1;

getSummaryModularity = 0;


plotMeanModularity = 0;
plotModuleHist = 0;
plotDasguptaHist = 0;
plotMeanStability = 0;

plotHalfLifeMeasures = 0;
plotModuleStability2 = 0;
    calcStability = 1;
plotCorrModuleEmbedding = 0;
plotHierarchyTime = 0;
plotTreeSimilarity = 0;
plotDatasetStability = 0;
    calcStabilityDatasets = 1;
plotDatasetStability2 = 0;
    calcStabilityDatasets = 1;
plotStabilityPerCut = 0;
    calcStabilityCuts = 1;

% examples
plotModuleSeqs = 0;
plotModuleDendo = 1;
plotExampleModules = 0;
    plotExamplesRandom = 0;
plotExampleTransition2Modularity = 0;
plotModuleVids = 0;

%% paths
if trainData
    mpath = [anadir '/modularity_train'];
else
    mpath = [anadir '/modularity_test'];
end

if ~exist(mpath); mkdir(mpath); end

figdir = [mpath '/Figures2b'];
if ~exist(figdir); mkdir(figdir); end

%% settings
fs = 30;

nrand = 10; % 56
%lags = [1:1:49, 50:2:100, 110:10:1000];
%lags = [1:1:49, 50:5:100, 110:10:200, 300:100:1000];
lags = [1:1:49, 50:5:100, 110:10:200, 300:100:1000];
%lags = [1 100];

%% prep things
%{
if max(idat)==18 % orig data
    s = [get_code_path() '/bhv_cluster/NHP_big_cage_data_log.xlsx'];
    useOldData = 1;
elseif max(idat)==87 %all yoda ephys
    s = [get_code_path() '/bhv_cluster/data_log_yoda_ephys2.xlsx'];
    useOldData = 0;
elseif max(idat)==88 %all yoda ephys, final
    s = [get_code_path() '/bhv_cluster/data_log_yoda_ephys_bv.xlsx'];
    useOldData = 0;
elseif max(idat)==22 %just yoda may ephys
    s = [get_code_path() '/bhv_cluster/data_log_yoda_ephys_may.xlsx'];
    useOldData = 0;
else
    error('which data?')
end
%}

if strcmp(thisMonk,'wo')
    s = [get_code_path() '/bhv_cluster/data_log_woodstock_ephys.xlsx'];
    useOldData = 1;
elseif strcmp(thisMonk,'yo')
    s = [get_code_path() '/bhv_cluster/data_log_yoda_ephys.xlsx'];
    useOldData = 1;
end

taskInfo = readtable(s);
%datasets = taskInfo.name;
names = cellfun(@(x) x(1:end-9), {datasets.name}, 'un',0);
[~,is] = ismember(names,taskInfo.name);
taskInfo = taskInfo(is,:);


monks = cellfun(@(x) x(1:2), unique(taskInfo.ck), 'un',0);
%tasks = strrep( strrep(unique(taskInfo.Condition),' =',''), '_',' ');
tasks = unique(taskInfo.Condition);


% state based information
istate = find(diff(C)~=0);
C_state = C(istate);

nstate = max(C_state);

idat_state = idat(istate);

if 0
    st = [1; istate];
    fn = [istate+1; numel(C)];
    Y_state = nan(numel(st),2);
    for is=1:numel(st)
        tmp = Y(st(is):fn(is),:);
        Y_state(is,:) = mean(tmp,1);
    end
end

%% calculations
if calcModularityHierarchy

    % states that arent in datasets
    ignoredStates = {};
    for id=1:max(idat)
        sel = idat(istate)==id;
        c = C_state(sel);
        n = accumarray(c,1,[nstate 1]);
        bad = find(n==0);
        good = find(n~=0);
        idx = [good; bad];

        ignoredStates{id,1} = idx;
        ignoredStates{id,2} = bad;
        ignoredStates{id,3} = good;
    end

    sname = [mpath '/ignoredStates.mat'];
    save(sname,'ignoredStates')

    % ------------------------------------------------------------
    % observed
    fprintf('getting all transitions')
    nsmp = numel(lags)*max(idat);
    %PO = nan(nstate,nstate,nsmp);
    PO = cell(1,nsmp);
    ii = 0;
    for id=1:max(idat)
        for ilag = 1:numel(lags)
            dotdotdot(ii,0.1,nsmp)
            sel = idat(istate)==id;
            c = C_state(sel);
            thisLag = lags(ilag);

            po = calculate_cond_trans_prob(c, [],nstate,thisLag);

            % deal with non-existant states in this data
            good = ignoredStates{id,3};
            po = po(good,good);

            % store
            ii = ii+1;
            PO{ii} = po;
        end
    end
    fprintf('\n')

    % test
    if 0
    po = cat(3,PO(:,:,1),PO(:,:,1));
    po = po(1:end-4,1:end-4,:);

    dataname = [mpath '/po_test'];
    call_cluster_hier(po,mpath,dataname)
    tmp2 = load('po_test_out.mat');
    end

    % cluster
    if doClustering
        dataname = [mpath '/po_obs'];
        call_cluster(PO,mpath,dataname)
    end

    if doHierarchicalClustering
        dataname = [mpath '/po_obs_hier'];
        call_cluster_hier(PO,mpath,dataname)
    end

    % ------------------------------------------------------------
    % random
    fprintf('getting all random transitions')
    nsmp = numel(lags)*max(idat)*nrand;
    PO = cell(1,nsmp);
    ii = 0;

    for ir=1:nrand
        for id=1:max(idat)
            sel = idat(istate)==id;
            c = C_state(sel);
            c = c(randperm(numel(c)));

            for ilag = 1:numel(lags)
                dotdotdot(ii,0.1,nsmp)
                thisLag = lags(ilag);

                po = calculate_cond_trans_prob(c, [],nstate,thisLag);

                % deal with non-existant states in this data
                good = ignoredStates{id,3};
                po = po(good,good);
                if sum(po(:))==0; eror('sup'); end
                ii = ii+1;
                PO{ii} = po;
            end
        end
    end
    fprintf('\n')

    if doClustering
        dataname = [mpath '/po_rand'];
        call_cluster(PO,mpath,dataname)
    end

    if doHierarchicalClustering
        dataname = [mpath '/po_rand_hier'];
        call_cluster_hier(PO,mpath,dataname)
    end

    foo=1;
end

%% load and reformat data
if prepData
    sname = [mpath '/modularity_train.mat'];

    if ~loadPreppedData
        fprintf('prepping data...\n')

        tic
        load([mpath '/ignoredStates.mat'])

        % ------------------------------------------------------------
        % clustering
        % load
        fprintf('loading back...\n')
        load([mpath '/ignoredStates.mat']);
        data_obs = load([mpath '/po_obs_out.mat']);
        data_rand = load([mpath '/po_rand_out.mat']);
        data_obs_in = load([mpath '/po_obs_in.mat']);
        data_rand_in = load([mpath '/po_rand_in.mat']);

        needsReformat = ~iscell(data_obs.labels);
        
        % make sure format is right
        if needsReformat
            lab = cell(1,size(data_obs.labels,1));
            labr = cell(1,size(data_rand.labels,1));

            for ii=1:numel(lab)
                lab{ii} = data_obs.labels(ii,:);
            end
            for ii=1:numel(labr)
                labr{ii} = data_rand.labels(ii,:);
            end
            data_obs.labels = lab;
            data_rand.labels = labr;
        end
        
        % re-jigger ignored states
        fprintf('\tre-imputing ignored states...\n')

        %error('gotta reformat the states according to the ignore ones!')
        PO_in = reshape(data_obs_in.po,[numel(lags) max(idat)]);
        PO_in_rand = reshape(data_rand_in.po,[numel(lags) max(idat) nrand]);
        
        Qlab = reshape(data_obs.labels,numel(lags),max(idat));
        Qrlab = reshape(data_rand.labels,numel(lags),max(idat),nrand);
            

        for id=1:max(idat)
            for ilag=1:numel(lags)
                good = ignoredStates{id,3};

                % prob
                tmp = PO_in{ilag,id};
                tmp2 = nan(nstate,nstate);
                tmp2(good,good) = tmp;
                PO_in{ilag,id} = tmp2;

                % labels
                q = nan(1,nstate);
                q(good) = Qlab{ilag,id};
                %q(isnan(q)) = max(q)+1;
                Qlab{ilag,id} = q;

                for ir=1:nrand
                    % prob
                    tmp = PO_in_rand{ilag,id,ir};
                    tmp2 = nan(nstate,nstate);
                    tmp2(good,good) = tmp;
                    PO_in_rand{ilag,id,ir} = tmp2;

                    % labels
                    qr = nan(1,nstate);
                    qr(good) = Qrlab{ilag,id,ir};
                    Qrlab{ilag,id,ir} = qr;
                end
            end
        end
        data_obs_in.po = cat(3,PO_in{:});
        data_rand_in.po = cat(3,PO_in_rand{:});
        data_obs.labels = cat(3,Qlab{:});
        data_rand.labels = cat(3,Qrlab{:});


        % reformat
        PO_in = reshape(data_obs_in.po,[nstate nstate,numel(lags),max(idat)]);
        PO_in_rand = reshape(data_rand_in.po,[nstate nstate,numel(lags),max(idat),nrand]);

        Q = data_obs.modularity;
        Q = reshape(Q,numel(lags),max(idat));
        Qlab = data_obs.labels;
        Qlab = permute( reshape(Qlab,nstate,numel(lags),max(idat)), [2 3 1]);

        Qr = data_rand.modularity;
        Qr = reshape(Qr,numel(lags),max(idat),nrand);
        Qrlab = data_rand.labels;
        Qrlab = permute( reshape(Qrlab,nstate,numel(lags),max(idat),nrand), [2 3 4 1]);


        % ------------------------------------------------------------
        % hieararchihcal clustering
        % load
        fprintf('\tloading hierarchichal clustering...\n')
        data_obs_hier = load([mpath '/po_obs_hier_out.mat']);
        data_rand_hier = load([mpath '/po_rand_hier_out.mat']);

        if needsReformat
            d = cell(1,size(data_obs_hier.dendogram,1));
            dr = cell(1,size(data_rand_hier.dendogram,1));

            for ii=1:numel(d)
                d{ii} = squeeze(data_obs_hier.dendogram(ii,:,:));
            end
            for ii=1:numel(dr)
                dr{ii} = squeeze(data_rand_hier.dendogram(ii,:,:));
            end
            data_obs_hier.dendogram = d;
            data_rand_hier.dendogram = dr;
        end
        
        % reformat
        TDS = reshape(data_obs_hier.tree_sampling_divergence,numel(lags),max(idat));
        DGS = reshape(data_obs_hier.dasgupta_score,numel(lags),max(idat));
        TDSr = reshape(data_rand_hier.tree_sampling_divergence,numel(lags),max(idat),nrand);
        DGSr = reshape(data_rand_hier.dasgupta_score,numel(lags),max(idat),nrand);


        % OBS, get modularity for diff cuts of the dendrogram
        fprintf('\tobserved modularity for diff cuts of OBS transitions')
        tmph = reshape(data_obs_hier.dendogram,numel(lags),max(idat));
        tmphr = reshape(data_rand_hier.dendogram,numel(lags),max(idat),nrand);

        if 0
            mn1 = min(min(cellfun(@(x) min(x(x(:,3)~=Inf,3)),tmph)));
            mx1 = max(max(cellfun(@(x) max(x(x(:,3)~=Inf,3)),tmph)));
            mn2 = min(min(min(cellfun(@(x) min(x(x(:,3)~=Inf,3)),tmphr))));
            mx2 = max(max(max(cellfun(@(x) max(x(x(:,3)~=Inf,3)),tmphr))));
            %lim = [min([mn1 mn2]) max([mx1 mx2])];
            lim = [min([mn1 mn2]) max([mx1 min(mx2,2*mx1)])];
            lim = ceil(lim*100)/100;
        else
            %lim = [0.4 1.3];
            %lim = [1.1 1.2];
            lim = [0.2 1.5];
        end

        % use cutoffs, or nclusts?
        if useCutoffs
            Cutoffs = lim(1):0.05:lim(2);
        else
            Cutoffs = 2:1:nstate-1;
        end
        ncuts = numel(Cutoffs);

        sz = size(tmph);
        Qh = nan([sz(1:2) ncuts]);
        Qh_lab = nan(size(Qlab));
        Qh_lab_all = nan([size(Qlab),ncuts]);
        Qh_mx = nan(size(Q));
        for ilag=1:numel(lags)
            dotdotdot(ilag,0.1,numel(lags))
            for id = 1:max(idat)
                % prep tree
                z = tmph{ilag,id};
                z = z(:,1:3);
                z(:,1:2) = z(:,1:2)+1;
                z(z==Inf) = nan; %max(z(:,3));

                % prep transitions
                po = PO_in(:,:,ilag,id);
                good = ignoredStates{id,3};
                po = po(good,good);

                tmplab = nan(numel(Cutoffs),nstate);
                for ic=1:numel(Cutoffs)
                    % cluster
                    if useCutoffs
                        c = cluster(z,'Cutoff',Cutoffs(ic),'Criterion','distance');
                    else
                        if Cutoffs(ic) > numel(good) % dont even try, not enough included states
                            continue
                        end
                        c = cluster(z,'maxclust',Cutoffs(ic));
                    end
                    q = modularity(po, c);
                    Qh(ilag,id,ic) = q;

                    % save hier cuts for later
                    tmp = nan(1,nstate);
                    tmp(good) = c-1; % be consistent
                    %tmplab(ic-1,:) = tmp;
                    tmplab(ic,:) = tmp;
                    foo=1;
                end

                % save clustering for max modularity
                q = squeeze(Qh(ilag,id,:));
                [mx,imx] = max(q);
                Qh_lab(ilag,id,:) = tmplab(imx,:);
                Qh_mx(ilag,id) = mx;

                Qh_lab_all(ilag,id,:,:) = tmplab';

                foo=1;
            end
        end

        % RAND, get modularity for diff cuts of the dendrogram
        fprintf('\tmodularity for diff cuts of RAND transitions')

        sz = size(tmphr);
        Qhr = nan([sz(1:2) ncuts nrand]);
        Qhr_lab = nan(size(Qrlab));
        Qhr_lab_all = nan([size(Qrlab), ncuts]);
        Qhr_mx = nan(size(Qr));

        for ir=1:nrand
            dotdotdot(ir,0.1,nrand)
            for ilag=1:numel(lags)
                for id = 1:max(idat)
                    % prep tree
                    z = tmphr{ilag,id,ir};
                    z = z(:,1:3);
                    z(:,1:2) = z(:,1:2)+1;
                    z(z==Inf) = nan; %max(z(:,3));

                    % prep transitions
                    po = PO_in_rand(:,:,ilag,id,ir);
                    good = ignoredStates{id,3};
                    po = po(good,good);

                    % modularity for diff cuts
                    tmplab = nan(numel(Cutoffs),nstate);
                    for ic=1:numel(Cutoffs)
                        if useCutoffs
                            c = cluster(z,'Cutoff',Cutoffs(ic),'Criterion','distance');
                        else
                            if Cutoffs(ic) > numel(good) % dont even try, not enough included states
                                continue
                            end
                            c = cluster(z,'maxclust',Cutoffs(ic));
                        end

                        q = modularity(po, c);
                        Qhr(ilag,id,ic,ir) = q;

                        % save for first cut
                        tmp = nan(1,nstate);
                        tmp(good) = c-1; % be consistent
                        tmplab(ic,:) = tmp;

                        foo=1;
                    end


                    % save clustering for max modularity
                    q = squeeze(Qhr(ilag,id,:,ir));
                    [mx,imx] = max(q);
                    Qhr_lab(ilag,id,ir,:) = tmplab(imx,:);
                    Qhr_mx(ilag,id,ir) = mx;

                    Qhr_lab_all(ilag,id,ir,:,:) = tmplab';

                end
            end
        end

        toc

        % save
        fprintf('\t saving...\n')
        %save(sname,'Q','Qr','Qh','Qlab','Qrlab','Qh_lab','Qhr_lab','Qh_mx','Qhr_mx','Qhr','Qhr_lab_all','PO_in','DGS','DGSr','TDS','TDSr')
        trans_lags = lags;
        save(sname,'-v7.3','trans_lags','Cutoffs','Q','Qr','Qh','Qlab','Qrlab','Qh_lab','Qh_lab_all','Qhr_lab','Qh_mx','Qhr_mx','Qhr','Qhr_lab_all','PO_in','DGS','DGSr','TDS','TDSr')
    else
        fprintf('loading data...\n')
        load(sname)
    end

    % maybe useful for plotting
    if 0
        error('rejigger for ignored states!')
        dendo = reshape(data_obs_hier.dendogram,numel(lags),max(idat),nstate-1,4);
        dendo(:,:,:,1:2) = dendo(:,:,:,1:2) + 1;
        dendo(dendo==Inf) = nan; % for non-sampled poses
        %dendo(isnan(dendo)) = max(dendo(:))*1.1;
        dendor = reshape(data_rand_hier.dendogram,numel(lags),max(idat),nrand,nstate-1,4);
        dendor(:,:,:,:,1:2) = dendor(:,:,:,:,1:2) + 1;
        dendor(dendor==Inf) = nan; % for non-sampled poses
        %dendor(isnan(dendor)) = max(dendor(:))*1.1;
    end

    % tmp save
    Q_orig = Q;
    Qr_orig = Qr;
    Qlab_orig = Qlab;
    Qrlab_orig = Qrlab;

end

% choose what to use for stuff
if useHierResults % from the first cut of the tree
    
    Q = Qh_mx;
    Qr = Qhr_mx;
    Qlab = Qh_lab;
    Qrlab = Qhr_lab;
    
    %Q = Qh(:,:,9);
    %Qr = squeeze(Qhr(:,:,9,:));
    %Qlab = Qh_lab_all(:,:,:,9);
    %Qrlab = squeeze(Qhr_lab(:,:,:,9,:));
else
    Q = Q_orig;
    Qr = Qr_orig;
    Qlab = Qlab_orig;
    Qrlab = Qrlab_orig;
end

%% summary of modularity stuff
if getSummaryModularity
    ilag = 1;

    % get module id
    C_mod = nan(size(idat));
    nModules = [];
    moduleCount = {};
    for id=1:max(idat)
        sel = idat==id;
        tmpc = C(sel);

        % module
        oldval = 1:nstate;
        newval = squeeze(Qlab(ilag,id,:))+1;
        tmpc = changem(tmpc,newval,oldval);
            
        [tmpc2,tmps] = clean_states(tmpc,minSmp,0.001);

        C_mod(sel) = tmpc;
        nModules(id) = max(newval);
    end

    % save
    sname = [mpath '/C_mod_train.mat'];
    save(sname,'C','C_mod')


    % duration of modules
    imod = find(diff(C_mod)~=0);
    mod_dur = [imod(1); diff(imod)] ./ fs;
    iC_mod = [C_mod(1); C_mod(imod+1)];
    id2 = [idat(1); idat(imod)];

    n = histcounts2(iC_mod,id2);
    m = accumarray([iC_mod,id2],1);

    [mu,se] = avganderror(nModules,'mean');
    [mud,sed] = avganderror(mod_dur,'mean');

    fprintf('mean nModule=%.3g + %.3g\nmean dur=%.3g+%.3g sec\nrange: %.3g, %.3g\n',...
        mu,se,mud,sed,min(mod_dur),max(mod_dur));

    % duration of each indiv module
    tmp = nan(max(idat),nstate);
    for id=1:max(idat)
        a = C_mod(idat==id);
        idx = find(diff(a)~=0);
        d = [diff(idx); numel(a)-idx(end)+1] ./ fs;
        ic = a(idx);

        n = accumarray(ic,1,[nstate 1]);
        s = accumarray(ic,d,[nstate 1]);
        p = s ./ n;

        tmp(id,:) = p;
    end

    fprintf('\n range of mean module dur: %.3g %.3g\n',min(tmp(:)), max(tmp(:)));

%     n = histcounts2(C_
    foo=1;
end

%% plot sequences of each module
if plotModuleSeqs
    isVis = 'off';

    ilag = 1;
    nplot = 100;
    timeLim = [1 4]*fs;
    dt = 10; % frame step

    % get module id
    C_mod = nan(size(idat));
    for id=1%:max(idat)
        fprintf('data %g: ',id)
        sel = find(idat==id);
        %itmp = istate(sel);
        tmpc = C(sel);

        % module
        oldval = 1:nstate;
        newval = squeeze(Qlab(ilag,id,:))+1;
        tmpm = changem(tmpc,newval,oldval);

        % for each module, get the modal sequence
        for ic=1:max(newval)
            fprintf(' %g,',ic)
            sel2 = tmpm==ic;
            [st,fn] = find_borders(sel2);

            % only plot some
            d = fn-st+1;
            good = find(d >= timeLim(1) & d <= timeLim(2))';
            %good = find(d >= 3 & d <= 10)';
            %if numel(good) > nplot; idx = randperm(numel(good),nplot);
            %else; idx = 1:numel(good);
            %end
            %plotThese = good(idx);
            plotThese = good;
            for is=plotThese
                %idx = itmp(st(is):fn(is));
                idx = sel(st(is):dt:fn(is));
                tmpx = X_notran(idx,:);
                tmpcom = com(idx,:);
                tmpc2 = tmpc(st(is):fn(is));

                % plot these states, in original space
                figure('visible',isVis)
                nr = 2; nc = size(tmpx,1);
                cols = colormap(copper(size(tmpx,1)));
                for ip=1:nc
                    % pose
                    subplot(nr,nc,ip)
                    plot_monkey_slim({tmpx,labels},[],ip);
                    set(gca,'xticklabel',[],'yticklabel',[],'zticklabel',[])
                    str = sprintf('%g',tmpc2(ip));
                    ylabel(str)
                    set(gca,'visible','off')

                    % marker
                    if 1
                    subplot(nr,nc,1+nc)
                    plot3(tmpcom(ip,1),tmpcom(ip,3),tmpcom(ip,2),...
                        '.','markersize',10,'color',cols(ip,:))

                    hold all
                    ax = gca;
                    end
                end

                % finish the position plot
                if 1
                subplot(nr,nc,1+nc)
                lim = [-5 5; -5 5; -2 6];
                set(ax,'xlim',[-5 5],'ylim',[-5 5],'zlim',[-2 6])
                grid on
                axis square

                sh = ones(nc,1);
                c = [0.8 0.8 0.8];
                plot3(tmpcom(:,1),tmpcom(:,3),lim(3,1)*sh,'.','color',c,'markersize',10);
                plot3(tmpcom(:,1),lim(2,2)*sh,tmpcom(:,2),'.','color',c,'markersize',10);
                plot3(lim(1,2)*sh,tmpcom(:,3),tmpcom(:,2),'.','color',c,'markersize',10);
                end

                % finish
                tightfig
                set_bigfig(gcf,[1 0.3])
                drawnow

                % save
                if saveFig
                    figdir2 = sprintf('%s/module_seqs/data%g_seq%g',...
                        figdir,id,ic);
                    if ~exist(figdir2); mkdir(figdir2); end
                    sname = sprintf('%s/%g',figdir2,is);
                    %save2pdf(sname);
                    str = sprintf('print -painters -dsvg %s',sname);
                    eval(str)
                end
                close(gcf)
            end
        end
        fprintf('\n')
        
        % colorbar
        figure
        colormap(copper)
        colorbar
        
        sname = sprintf('%s/module_seqs/colorbar',figdir);
        save2pdf(sname)
        close(gcf)
    end

    % for each dataset

    foo=1;
    % control: sequence ID as a function of height
    %{
    Z = com(:,2);
    zlim = [0 max(Z)];
    zbin = linspace(zlim(1),zlim(2),10);

    RES = {};
    for id=1:max(idat)
        c1 = C_mod(idat==id);
        nmod = max(c1);
        P = [];
        for iz = 1:numel(zbin)-1
            sel = idat==id & (Z>=zbin(iz) & Z<= zbin(iz+1));
            c2 = C_mod(sel,:);
            n = accumarray(c2,1,[nmod 1]);
            %p = n ./ sum(n);
            p = n ./ numel(c1);
            P(iz,:) = p;
        end
        RES{id} = P;
    end

    % plot all
    figure('name','module at heights')
    [nr,nc] = subplot_ratio(max(idat));
    for id=1:max(idat)
        p = RES{id};
        z = mean([zbin(1:end-1); zbin(2:end)]);

        subplot(nr,nc,id)
        imagesc(1:size(p,2),z,p)
        set(gca,'ydir','normal')
        s = sprintf('id=%g',id);
        title(s)
    end
    set_bigfig
    %}
end


%% plot each individual dendrogram
if plotModuleDendo
    isVis = 1;
    plotPoses = 0;

    ilag = 1;
    % 11, 13, 17
    fprintf('plotting caldograms: ')
    for id=1:max(idat) %[11 13 17]%
        fprintf('%g,',id)
        tmph = reshape(data_obs_hier.dendogram,numel(lags),max(idat));

        T = tmph{ilag,id}(:,1:3);
        T(:,1:2) = T(:,1:2)+1;

        % get labels
        %lab = squeeze(Qlab(ilag,id,:))+1;
        lab = squeeze(Qlab(ilag,id,:))+1;
        bad = isnan(lab);
        lab(bad) = [];

        %https://www.mathworks.com/matlabcentral/answers/324932-dendrogram-with-custom-colouring
        nModule   = max(lab);
        cutoff      = median([T(end-nModule+1,3) T(end-nModule+2, 3)]);

        map = [find(~bad), [1:numel(lab)]'];

        % plot
        if plotPoses
            out = plot_cladogram(T,pose_mu,map,nModule,labels,isVis,plotPoses);
        else
            out = plot_cladogram(T,[],map,nModule,[],isVis,plotPoses);
        end
        
        % reorder leaf IDs
        axden = out.ax_den;

        xt = str2num(get(axden,'xticklabel'));
        m = map(xt,1);
        m = cellfun(@num2str,num2cell(m),'un',0);
        set(axden,'xticklabel',m)

        % plot module ID, for reference
        if 0
            tmp = lab(xt);
            for ic=1:max(lab)
                sel=tmp==ic;
                tmpx = mean(find(sel));
                tmpy = max(get(axden,'ylim'));
                %col = unique(cat(1,out.color_map{sel,1}),'rows');
                col = 'k';
                text(axden,tmpx,tmpy,num2str(ic),'color',col)
            end
        end

        tmp = lab(xt);
        theseLeaves = [1; find(diff(tmp)~=0)+1];
        hl = legend(out.hleaf(theseLeaves),cellfun(@num2str,num2cell(tmp(theseLeaves)),'un',0),'location','eastoutside');
        hl.Title.String = 'Module';

        % allow for log spacing, get rid of 0s
        y = get(out.hden,'ydata');
        y = cat(1,y{:});
        mn = min(y(y>0));

        y(y==0) = mn *0.8;

        for ih=1:numel(out.hden)
            set(out.hden(ih),'ydata',y(ih,:));
        end

        set(axden,'yscale','log')
        set(axden,'TickLength',[0.005 0.002])

        tstr = sprintf('%s: %s,%s\nnModule=%g',...
            taskInfo.name{id},taskInfo.ck{id},taskInfo.Condition{id},nModule);
        title(axden,tstr)

        % save
        if saveFig
            figdir2 = [figdir '/cladogram_module'];
            if ~exist(figdir2); mkdir(figdir2); end
            sname = sprintf('%s/clado_id%g',figdir2,id);
            save2pdf(sname,gcf)
            %str = sprintf('print -painters -dsvg %s',sname);
            %eval(str)
            close(gcf)
        end
    end
    fprintf('\n')
end



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
    set_bigfig(gcf,[0.25 0.35])

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

    % ------------------------------------
    % histogram for lag==1
    ilag = 1;

    % get means
    q = Q(ilag,:);
    qr = Qr(ilag,:,:);

    % by subject/task
    G = {taskInfo.ck, taskInfo.Condition};
    [P,T,STATS,TERMS]=anovan(q,G,'display','off');

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

    str = sprintf('Modularity Scores\npop-p=%.3g\n num sign sets: %g/%g\nAnovan, monk: F=%.3g, p=%.3g\ntask: F=%.3g, p=%.3g',...
        p,sum(pi<0.05),numel(pi),T{2,6},T{2,7},T{3,6},T{3,7});
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
    set_bigfig(gcf,[0.5 0.5])

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

    % subject/task
    G = {taskInfo.ck, taskInfo.Condition};
    [P,T,STATS,TERMS]=anovan(q,G,'display','off');

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

    str = sprintf('Dasgupta Scores\npop-p=%.3g\n num sign sets: %g/%g\nAnovan, monk: F=%.3g, p=%.3g\ntask: F=%.3g, p=%.3g',...
        p,sum(pi<0.05),numel(pi),T{2,6},T{2,7},T{3,6},T{3,7});
    title(str)
    xlabel('Dasgupta Score')
    ylabel('Proportion')
    axis square

    if saveFig
        sname = [figdir '/hist_dasgupta_lag' num2str(lags(ilag))];
        save2pdf(sname,gcf)
    end
end



%% modularity half life by task, subject
if plotHalfLifeMeasures

    %figure; [nr,nc] = subplot_ratio(18);

    mstrs = {'Modularity','Dasgupta','Stability'};
    for imeasure=[1 2 3] %1:3
        if imeasure==1
            thisMeasure = Q;
            thisX = lags;
        elseif imeasure==2
            thisMeasure = DGS;
            thisX = lags;
        else
            % observed
            thisMeasure = nan(numel(lags)-1,max(idat));
            irep = 0;
            for id=1:max(idat)
                for il1=1:numel(lags)-1
                    il2 = il1+1;
                    %q1 = squeeze(Qlab(il,id,:));
                    q1 = squeeze(Qlab(il1,id,:))+1;
                    q2 = squeeze(Qlab(il2,id,:))+1;

                    bad = isnan(q1);
                    mx = ami(q1(~bad),q2(~bad));


                    % store
                    thisMeasure(il1,id) = mx;
                    foo=1;
                end
            end
            thisX = lags(2:end);
        end


        %half life
        f = fittype('a*exp(b*x) + c');
        H = [];
        Hv = [];
        Qfit = [];
        R2 = [];

        for id=1:max(idat)
            a = thisMeasure(:,id);
            %interpn(lags,a,h,'spline')

            ub = [1 0 (max(a)-min(a))/2+min(a)];
            lb = [0 -10 0];
            st = [0.5 -0.01 min(a)];

            %[curve1, goodness] = fit(thisX',a,'exp2','upper',ub,'lower',lb,'startpoint',st);
            %[curve, g] = fit(thisX',a,'exp2');
            %[curve,g] = fit(thisX',a,f);
            [curve,g] = fit(thisX',a,f,'upper',ub,'lower',lb,'startpoint',st);

            %subplot(nr,nc,id); plot(lags,a); hold all; plot(curve); set(gca,'xscale','log'); title(num2str(id))
            %figure; plot(thisX,a); hold all; plot(curve); set(gca,'xscale','log'); title(num2str(id))

            %[curve,g] = fit(thisX',a-min(a),'exp1');
            %figure; plot(thisX,a-min(a)); hold all; plot(curve); set(gca,'xscale','log'); title(num2str(id))

            %tmp = [log(2)./abs(curve.b), log(2)./abs(curve.d)];
            %tmp = sort(tmp);
            tmp = [-log(2)./curve.b 0];

            Qfit(:,id) = curve(thisX);
            H(id,:) = tmp;
            %H(id,1) = log(2)./abs(curve.b);
            %H(id,2) = log(2)./abs(curve.d);
            Hv(id,:) = curve(H(id,:));
            R2(id) = g.adjrsquare;
        end

        % start figure
        figure
        nr = 1; nc = 2;
        %set_bigfig(gcf,[0.9 0.6])
        set_bigfig(gcf,[0.6 0.4])

        hstr = {'fast','slow'};
        hax = [];
        for ih=1%:2
            H2 = H(:,ih);
            Hv2 = Hv(:,ih);

            % anova
            if useOldData
                itask = contains(taskInfo.Condition,'feeder')+1;
                tasks2 = {'noTask','feeder'};
                imonk = taskInfo.ck;
            elseif 0
                itask = ones(size(taskInfo,1),1);
                tasks2 = {'env'};
                imonk = taskInfo.ck;
            else
                [tasks2,~,itask] = unique(taskInfo.Condition);
                imonk = taskInfo.ck;
            end

            avgstr = 'mean';
            G = {imonk,itask};
            if strcmp(avgstr,'mean') % mean
                [P,T,STATS,TERMS]=anovan(H2,G,'display','off');
            else
                [~,tmph] = sort(H2);
                [P,T,STATS,TERMS]=anovan(tmph,G,'display','off');
            end

            % get means
            MU = [];
            SE = [];
            for im=1:numel(monks)
                for it = 1:numel(tasks2)
                    %sel = strncmp(taskInfo.name,monks{im},2) & ismember(taskInfo.Condition,tasks{it});
                    sel = strncmp(imonk,monks{im},2) & itask==it;
                    tmp = H2(sel);
                    [mu,se] = avganderror(tmp,avgstr,1,1,200);
                    MU(im,it) = mu;
                    SE(im,it) = se;
                end
            end

            % ---------------------------------------
            % plot

            % all curves
            subplot(nr,nc,1+(ih-1)*nc)
            ln = {'-','--'};
            mk = {'o','^'};
            cols = get_safe_colors(0,[1 2 4]);
            h = nan(1,max(idat));
            for id=1:size(Q,2)
                selm = strncmp(monks,taskInfo.name{id},2);
                %selt = ismember(tasks,taskInfo.Condition{id});
                selt = itask(id);
                tmp = Qfit(:,id);
                h(id) = plot(thisX,tmp,ln{selm},'color',cols(selt,:),'linewidth',2);
                hold all
                plot(H2(id),Hv2(id),['k' mk{selm}],'markerfacecolor',cols(selt,:),'markersize',7)
                set(gca,'ylim',[0 max(Qfit(:))*1.05])
                %plotcueline('x',H2(id),mk{selm},'color',cols(selt,:))
                foo=1;
            end

            [mr,sr] = avganderror(R2,'mean');
            tstr = sprintf('%s half life, %s\nR2=%.3g + %3.g',mstrs{imeasure},hstr{ih},mr,sr);
            title(tstr)
            xlabel('# transitions')
            ylabel(['Mean ' mstrs{imeasure}])
            set(gca,'xscale','log','xlim',[0 1000],'ylim',[max(min(Qfit(:)), 0) max(get(gca,'ylim'))])

            axis square
            lstr = cellfun(@(x,y) [x(1:2) '-' y],taskInfo.name,tasks2(itask)','un',0);
            legend(h,lstr,'location','eastoutside')

            hax(ih,1) = gca;

            % mean half life
            subplot(nr,nc,2+(ih-1)*nc)
            hb = barwitherr(SE,MU);
            for ii=1:numel(hb)
                set(hb(ii),'facecolor',cols(ii,:))
            end

            s = sprintf('%s half life\nAnovan, monk: F=%.3g, p=%.3g\ntask: F=%.3g, p=%.3g',...
                hstr{ih},T{2,6},T{2,7},T{3,6},T{3,7});
            title(s)
            ylabel('mean half life transition')
            xlabel('monk')

            set(gca,'xticklabel',monks)
            legend(tasks2,'location','eastoutside')
            axis square
            %set(gca,'yscale','log','ytick',0:10:100)

            hax(ih,2) = gca;
        end

        % adjust sizing
        drawnow
        pos = get(hax(1),'position');
        pos2 = get(hax(2),'position');
        pos2 = [pos2(1:2) pos(3:4)];
        set(hax(2),'position',pos2)


        % save
        if saveFig
            sname = [figdir '/halflife2_monkTask_' mstrs{imeasure}];
            save2pdf(sname,gcf)
        end
    end
end


%% module stability, comparing subejcts and tasks
if plotMeanStability

    meantype = 'median'; %mean

    thisStabilityScore = 3; %0=custom, 1=NMI, 2=rand index, 3=AMI
    tmp = {'custom','NMI','adjRAND','AMI'};
    stabStr = tmp{thisStabilityScore+1};

    labs = 1:nstate;

    if calcStability
        % observed
        S = nan(numel(lags)-1,max(idat));
        Sr = nan(numel(lags)-1,max(idat),nrand);

        irep = 0;
        nsmp = numel(S);
        fprintf('observed stability')
        for id=1:max(idat)
            for il1=1:numel(lags)-1
                irep=irep+1;
                dotdotdot(irep,0.1,nsmp)
                il2 = il1+1;

                %q1 = squeeze(Qlab(il,id,:));
                q1 = squeeze(Qlab(il1,id,:))+1;
                q2 = squeeze(Qlab(il2,id,:))+1;

                bad = isnan(q1);
                mx = ami(q1(~bad),q2(~bad));

                % store
                S(il1,id) = mx;
                foo=1;

               % random
               if 0
                good = find(~bad);
                qr2 = nan(size(q2));
                for ir=1:nrand
                    idx = good( randperm(numel(good)) );
                    qr2(good) = q2(idx);
                    mx = ami(q1(~bad),qr2(~bad));
                    Sr(il1,id,ir) = mx;
                end
               end
            end
        end
        fprintf('\n')

        % random
        if 1
            fprintf('rand stability')
            Sr = nan(numel(lags)-1,max(idat),nrand);
            nsmp = numel(Sr);
            irep = 0;
            for ir=1:nrand
                for id=1:max(idat)
                    for il1=1:numel(lags)-1
                        irep=irep+1;
                        dotdotdot(irep,0.1,nsmp)
                        il2 = il1+1;

                        q1 = squeeze(Qrlab(il1,id,ir,:))+1;
                        q2 = squeeze(Qrlab(il2,id,ir,:))+1;

                        bad = isnan(q1);
                        mx = ami(q1(~bad),q2(~bad));


                        % store
                        Sr(il1,id,ir) = mx;
                        foo=1;
                    end
                end
            end
        end

        % save
        %sname = [mpath '/modularity_stability_' stabStr '.mat'];
        %save(sname,'S','Sr')
        S_orig = S;
        Sr_orig = Sr;
    end
    fprintf('\n')

    S = S_orig;
    Sr = Sr_orig;

    %S(S<0) = 0;
    %Sr(Sr<0) = 0;

    %------------------------------------------------
    % plot
    figure
    nr = 1; nc = 1;
    set_bigfig(gcf,[0.4,0.4])

    % stability relative to some transition

    [mu,se] = avganderror(S,'mean',2);
    sr_mu = squeeze(nanmean(Sr,2));
    [mur,ser] = avganderror(sr_mu,'mean',2);
    xx = lags(2:end);

    %p = sum(sr_mu > mu,2) ./ size(Sr,4);
    tmpmu = nanmean(sr_mu,2);
    %error('change p-val')
    %p = sum(abs(sr_mu-tmpmu) > abs(mu-tmpmu),2) ./ size(sr_mu,2);
    
    p = sum(sr_mu > mu,2) ./ size(sr_mu,2);

    %p = sum(abs(sr_mu-mu) > 0,2) ./ size(sr_mu,2);
    %p = sum(sr_mu > mu | sr_mu < mu,2) ./ size(sr_mu,2);
    p = bonf_holm(p);
    mup = ones(size(xx)) * max(mu+se)*1.05;
    mup(p>0.05) = nan;

    % plot
   	figure
    shadedErrorBar(xx,mu,se,{'r.-'});
    hold all
    shadedErrorBar(xx,mur,ser,{'.-','color',ones(1,3)*0.5});
    plot(xx,mup,'k.-','linewidth',1)
    set(gca,'xscale','log')
    axis square

    title(['mean ' stabStr])
    xlabel('# transitions')
    ylabel([stabStr ' stability'])

    set_bigfig(gcf,[0.25 0.35])
    
    % save figure
    if saveFig
        sname = [figdir '/module_stability_' stabStr];
        save2pdf(sname,gcf)
    end
end

%% correlate embedding distance with module composition
if plotCorrModuleEmbedding
    nmax = 50;
    nboot = 5;

    D = [];
    fprintf('within vs between module distance')
    for ilag=1:numel(lags)
        dotdotdot(ilag,0.1,numel(lags))

        for id = 1:max(idat)
            q = squeeze(Qlab(ilag,id,:));

            tmpd = [];
            for ic=1:max(q)
                selw = find(q==ic-1);
                selw = find(ismember(C_state,selw) & idat_state==id);
                nw = numel(selw);

                selb = find(q ~= ic-1);
                selb = find(ismember(C_state,selb) & idat_state==id);
                nb = numel(selb);

                ib = 1;
                %for ib=1:nboot
                    % across module distance
                    %selb2 = selb( randperm(nb, min(nmax,nb)) );
                    selb2 = selb;
                    tmpb = Y_state(selb2,:);
                    db = sqrt( sum(sum((tmpb - mean(tmpb)).^2,2)) );
                    %db = mean(pdist(tmpb));

                    % within module distance
                    %selw2 = selw( randperm(nw, min(nw,nmax)) );
                    selw2 = selw;
                    tmpw = Y_state(selw2,:);
                    dw = sqrt( sum(sum((tmpw - mean(tmpw)).^2,2)) );
                    %dw = mean(pdist(tmpw));

                    % store
                    tmpd(1,ic,ib) = dw;
                    tmpd(2,ic,ib) = db;
                    foo=1;
                %end
            end
            D(ilag,id,:) = mean(mean(tmpd,3),2);

            foo = 1;
        end
    end

    % plot
    figure
    [mu,se] = avganderror(D,'mean',2);

    strs = {'within','between'};
    cols = get_safe_colors(0,1:2);
    h = [];
    for ii=1:2
        htmp = shadedErrorBar(lags,mu(:,:,ii),se(:,:,ii),{'.-','color',cols(ii,:)});
        h(ii) = htmp.mainLine;
        hold all
    end

    s = sprintf('within vs between module variance');
    title(s)
    xlabel('lags')
    ylabel('mean variance')

    set(gca,'xscale','log')
    legend(h,strs,'location','eastoutside')

    if saveFig
        sname = [figdir '/module_withinVSbet_var'];
        save2pdf(sname,gcf)
    end
end



%% time as a function of cluster number
if plotHierarchyTime
    ilag = 1;

    % get module id
    nModules = [];
    MOD_dur = [];
    MU=[];SE=[];
    for ic=1:nstate-1
        C_mod = nan(size(idat));
        for id=1:max(idat)
            sel = idat==id;
            tmpc = C(sel);

            % module
            oldval = 1:nstate;
            newval = squeeze(Qh_lab_all(ilag,id,:,ic))+1;
            tmpc = changem(tmpc,newval,oldval);
            C_mod(sel) = tmpc;
            nModules(id,ic) = max(newval);
        end

        % duration of modules
        imod = find(diff(C_mod)~=0);
        mod_dur = [imod(1); diff(imod)] ./ fs;

        % finish
        tmp = [mod_dur, ones(size(mod_dur))*ic];
        MOD_dur = cat(1,MOD_dur,tmp);
        [MU(ic),SE(ic)] = avganderror(mod_dur,'mean');
        foo=1;
    end

    % stats
    corrtype = 'spearman';
    [r,p] = corr(MOD_dur(:,1),MOD_dur(:,2),'type',corrtype);

    % plot
    figure
    shadedErrorBar(2:nstate,MU,SE)


    str = sprintf('cluster granularity vs mean time\n%s R=%.3g, p=%.3g',...
        corrtype,r,p);
    title(str)
    xlabel('Cluster Granularity')
    ylabel('Mean time (s)')

    % save
    if saveFig
        sname = [figdir '/corr_nclusterVStime_' corrtype];
        save2pdf(sname,gcf)
    end

end

%% simialrity between trees
if plotTreeSimilarity
    treedir = [mpath '/treeDist'];
    if ~exist(treedir); mkdir(treedir); end

    verbose = 1;

    ilag = 1;

    % save all trees
    disp('saving trees')
    tmph = reshape(data_obs_hier.dendogram,numel(lags),max(idat));
    treeNames = {};
    for id=1:max(idat)

        t=tmph{ilag,id}(:,1:3);
        t(:,1:2)=t(:,1:2)+1;
        T1=phytree(t);

        sname = sprintf('%s/tree%g_%g.tree',treedir,id,ilag);
        phytreewrite(sname, T1)
        treeNames{id} = sname;
    end

    % build list of comparisons
    cmb = combnk(1:max(idat),2);
    treeList ={};

    for ic=1:size(cmb,1)
        ii1 = cmb(ic,1);
        ii2 = cmb(ic,2);

        tmp = {treeNames{ii1},treeNames{ii2}};
        treeList = cat(1,treeList,tmp);
    end

    foo=1;

    % paths
    %rpath = 'Rscript';
    tmp = [get_code_path() '/bhv_cluster/matlab/R/call_treeDist.R'];
    [funcpath, func] = fileparts(tmp);

    % data
    dat = [];
    dat.treeList = treeList;

    % prepare file names
    tmpname = [treedir '/treeDist'];
    name_in = [tmpname '_in.mat'];
    name_out = [tmpname '_out.mat'];

    %save it
    save(name_in,'-struct','dat')

    % call the func
    commandStr = sprintf('cd %s; %s %s.R "%s"',funcpath,rpath,func,tmpname);

    if verbose
        [status,result] = system(commandStr,'-echo');
    else
        [status,result] = system(commandStr);
    end

    % load back
    out = load(name_out);
    toc

end


%% stability of modules as a function of hierarchy
if plotStabilityPerCut
    ilag = 1;
    avgtype = 'mean';
    removeMeans = 1;
    subtractRand = 0;
    
    if calcStabilityCuts
        nrand_s = 100;

        ndat = max(idat);
        ncut = numel(Cutoffs);
        
        Shc = nan(ndat,ndat,ncut);
        Shcr = nan(ndat,ndat,ncut,nrand_s);
        irep=0;
        nsmp = numel(Shc);
        for id1=1:ndat
            for id2=1:ndat
                for icut=1:ncut
                    if id1==id2; continue; end
                    irep=irep+1;

                    q1 = squeeze(Qh_lab_all(ilag,id1,:,icut))+1;
                    q2 = squeeze(Qh_lab_all(ilag,id2,:,icut))+1;

                    % stability
                    bad = isnan(q1) | isnan(q2);
                    mx = ami(q1(~bad),q2(~bad));

                    if isnan(mx)
                        foo=1;
                    end
                    % store
                    Shc(id1,id2,icut) = mx;

                    % random
                    if subtractRand
                        good = find(~bad);
                        qr2 = nan(size(q2));
                        for ir=1:nrand_s
                            idx = good( randperm(numel(good)) );
                            qr2(good) = q2(idx);
                            mx = ami(q1(~bad),qr2(~bad));
                            Shcr(id1,id2,icut,ir) = mx;
                        end
                    end
                end
            end
        end
        
        if subtractRand
            Shc = Shc - nanmean(Shcr,4);
        end
    end

    % prep
    [umonk,~,imonk] = unique(taskInfo.ck);
    umonk = cellfun(@(x) x(1:2),umonk,'un',0);
    if 0
        [utask,~,itask] = unique(taskInfo.Condition);
    else
        utask = {'noTask','task'};
        itask = contains(taskInfo.Condition,'feeder')+1;
    end
    %[ugrp,~,igrp] = unique([imonk,itask],'rows');

    % prep factors
    tmps = [];
    tmpsr = [];
    imonk2 = {};
    itask2 = {};

    cmb = nchoosek(1:size(taskInfo,1),2);
    for ic=1:size(cmb,1)
        ii1 = cmb(ic,1);
        ii2 = cmb(ic,2);
        for icut=1:numel(Cutoffs)
            tmps(ic,icut) = Shc(ii1,ii2,icut);
            tmpsr(ic,icut,:) = Shcr(ii1,ii2,icut,:);
        end
        imonk2{ic} = sprintf('%g-%g',imonk(ii1),imonk(ii2));
        itask2{ic} = sprintf('%g-%g',itask(ii1),itask(ii2));
    end
    imonk2(strcmp(imonk2,'2-1')) = {'1-2'};
    itask2(strcmp(itask2,'2-1')) = {'1-2'};
    umonk2 = {'1-1','2-2','1-2'};
    utask2 = {'1-1','2-2','1-2'};
    
    % select data
    if removeMeans
        for im=1:numel(umonk2)
            sel = contains(imonk2,umonk2{im});
            sel = sel & contains(itask2,{'1-1','2-2'});
            tmp = tmps(sel);
            tmps(sel,:) = tmps(sel,:) - nanmean(tmp);
        end
    end
    
    sel = contains(itask2,'1-1');
    tmp1 = tmps(sel,:);
    sel = contains(itask2,'2-2');
    tmp2 = tmps(sel,:);
    tmp = {tmp1,tmp2};

    % stats
    if 1
        if strcmp(avgtype,'mean')
            [~,p,~,s] = ttest2(tmp1,tmp2);
            s = s.tstat;
        else
            p=[];
            s=[];
            for icut=1:numel(Cutoffs)
                [p(icut),~,stmp] = ranksum(tmp1(:,icut),tmp2(:,icut));
                s(icut)=stmp.zval;
            end
        end
    else
        %[~,p,~,s] = ttest2(tmp1,tmp2);
        %stat = s.tstat;
        %[p,~,s] = ranksum_dim(1,tmp1,tmp2);
        %stat = [s.ranksum];
        stat = nanmean(tmp1)-nanmean(tmp2);
        
        nboot = 1000;
        tmp3 = [tmp1; tmp2];
        itmp = [ones(size(tmp1)); ones(size(tmp2))*2];
        n1 = size(tmp1,1);
        stat_boot = [];
        for ib=1:nboot
            idx = randperm(size(itmp,1),n1);
            t1 = tmp3(idx,:);
            t2 = tmp3;
            t2(idx,:)=[];
            stat_boot(ib,:) = nanmean(t1)-nanmean(t2);
            %[~,~,~,s] = ttest2(t1,t2);
            %stat_boot(ib,:) = s.tstat;
            %[~,~,s] = ranksum_dim(1,t1,t2);
            %stat_boot(ib,:) = [s.ranksum];
        end
        
        p = sum(abs(stat_boot) > abs(stat)) ./ nboot;
        foo=1;
    end
    %p = bonf_holm(p);
    
    % where is the mean of cutoffs?
    q = squeeze(Qh(ilag,:,:));
    [~,imx] = max(q,[],2);
    tmpc = Cutoffs(imx);
    [muc,sec] = avganderror(tmpc,'mean');
    
    
    % plot
    figure
    cols = get_safe_colors(0,[1 2]);
    h=[];
    mx=0;
    for ii=1:2
        t = tmp{ii};
        [mu,se] = avganderror(t,avgtype);
        htmp =shadedErrorBar(Cutoffs,mu,se,{'color',cols(ii,:)});
        h(ii)=htmp.mainLine;
        hold all
        mx = max(mx,max(mu+se));
    end
    
    mup = ones(size(p))*mx*1.02;
    mup(p>0.05) = nan;
    plot(Cutoffs,mup,'k.-','markersize',5)
    
    xx = [muc-sec, muc, muc+sec];
    yy = ones(size(xx))*mx*1.04;
    plot(xx,yy,'r.-')
    
    legend(h,utask,'location','eastoutside')
    xlabel('Cutoff')
    ylabel([avgtype ' Stability'])
    axis square
    set(gca,'xlim',[Cutoffs(1) Cutoffs(end)])
    set_bigfig(gcf,[0.25 0.35]);
    
    % save
    if saveFig
        sname = [figdir '/stabilityVScuts_taskVSnotask_' avgtype];
        save2pdf(sname,gcf)
    end
end

%% stability of modules between datasets
if plotDatasetStability2
    ilag = 1;
    avgtype = 'mean';
    removeMeans = 1;
    subtractRand = 0;

    
    if calcStabilityDatasets
        nrand_s = 100;

        ndat = max(idat);

        Sh = nan(ndat,ndat);
        Shr = nan(ndat,ndat,nrand_s);
        irep=0;
        nsmp = numel(Sh);
        for id1=1:ndat
            for id2=1:ndat
                if id1==id2; continue; end
                irep=irep+1;

                q1 = squeeze(Qlab(ilag,id1,:))+1;
                q2 = squeeze(Qlab(ilag,id2,:))+1;

                % stability
                bad = isnan(q1) | isnan(q2);
                mx = ami(q1(~bad),q2(~bad));

                if isnan(mx)
                    foo=1;
                end
                % store
                Sh(id1,id2) = mx;

                % random
                good = find(~bad);
                qr2 = nan(size(q2));
                for ir=1:nrand_s
                    idx = good( randperm(numel(good)) );
                    qr2(good) = q2(idx);
                    mx = ami(q1(~bad),qr2(~bad));
                    Shr(id1,id2,ir) = mx;
                end
            end
        end
        
        if subtractRand
            Sh = Sh - nanmean(Shr,3);
        end
    end

    % prep
    [umonk,~,imonk] = unique(taskInfo.ck);
    umonk = cellfun(@(x) x(1:2),umonk,'un',0);
    if 0
        [utask,~,itask] = unique(taskInfo.Condition);
    else
        utask = {'noTask','task'};
        itask = contains(taskInfo.Condition,'feeder')+1;
    end
    %[ugrp,~,igrp] = unique([imonk,itask],'rows');

     % prep factors
    tmps = [];
    tmpsr = [];
    imonk2 = {};
    itask2 = {};

    cmb = nchoosek(1:size(taskInfo,1),2);
    for ic=1:size(cmb,1)
        ii1 = cmb(ic,1);
        ii2 = cmb(ic,2);
        tmps(ic) = Sh(ii1,ii2);
        tmpsr(ic,:) = Shr(ii1,ii2,:);
        imonk2{ic} = sprintf('%g-%g',imonk(ii1),imonk(ii2));
        itask2{ic} = sprintf('%g-%g',itask(ii1),itask(ii2));
    end
    imonk2(strcmp(imonk2,'2-1')) = {'1-2'};
    itask2(strcmp(itask2,'2-1')) = {'1-2'};
    
    umonk2 = unique(imonk2);
    utask2 = unique(imonk2);
    
    % stats on similarity by subject and task
    G = {imonk2, itask2};
    if strcmp(avgtype,'mean')
        [P,T,STATS,TERMS]=anovan(tmps,G,'varnames',{'monk','task'},'display','off');
    else
        itmp = tiedrank(tmps);
        [P,T,STATS,TERMS]=anovan(itmp,G,'varnames',{'monk','task'},'display','off');
    end
    
    if saveFig
        sname = [figdir '/stability_anova_taskSubject.xlsx'];
        writecell(T,sname) % better to use full path
    end
    
    
     % overall structure?
    mu = nanmean(Sh(:));
    mur = squeeze(nanmean(nanmean(Shr),2));
    p = sum(mur > mu) ./ nrand_s;

    % get similarity between monkeys, tasks
    figure
    nr = 1; nc = 3;
    set_bigfig(gcf,[0.8 0.4])
    hax = [];

    % ===================================================
    % for all conditons
    MU = []; SE = [];
    MUr = []; SEr = [];
    [umonk3,~,imonk3] = unique(imonk2);
    [utask3,~,itask3] = unique(itask2);
    for im=1:numel(umonk3)
        for it=1:numel(utask3)
            sel = imonk3==im & itask3==it;

            % store
            tmp = tmps(sel);
            [MU(im,it),SE(im,it)] = avganderror(tmp(:),avgtype);

            for ir=1:nrand_s
                tmpr = tmpsr(sel,:);
                [MUr(im,it,ir),SEr(im,it,ir)] = avganderror(tmpr(:),avgtype);
            end
        end
    end

    pm = sum(MUr > MU,3) ./ nrand_s;
    pm = bonf_holm(pm);

    subplot(nr,nc,1)
    xx = 1:numel(utask3);
    yy = 1:numel(umonk3);
    imagesc(xx,yy,MU)

    hold all
    [ir,ic] = find(pm<0.05);
    plot(ic,ir,'w.','markersize',10)

    % finish
    str = sprintf('AMI similarity\nAnovan, monk: F=%.3g, p=%.3g\ntask: F=%.3g, p=%.3g',...
        T{2,6},T{2,7},T{3,6},T{3,7});
    title(str)

    xlab = cellfun(@(x) [utask{str2num(x(1))} '-' utask{str2num(x(3))}],utask3,'un',0);
    ylab = cellfun(@(x) [umonk{str2num(x(1))} '-' umonk{str2num(x(3))}],umonk3,'un',0);

    axis square
    set(gca,'xtick',xx,'xticklabel',xlab,'ytick',yy,'yticklabel',ylab)
    colorbar
    %set(gca,'clim',[0 max(get(gca,'clim'))])
    colormap(copper)
    
    hax(1) = gca;
    
    % ===================================================
    % within vs between subject
    grp = {{'1-1','2-2'},{'1-2'}};
    selg = contains(imonk2,[grp{:}]);
    
    
    tmps2 = tmps(selg);
    igrp = contains(imonk2(selg),grp{2})+1;
    itask_tmp = itask2(selg);
    
    if removeMeans
        for it=1:numel(utask2)
            selt = contains(itask_tmp,utask2{it});
            if strcmp(avgtype,'mean')
                tmps2(selt) = tmps2(selt) - nanmean(tmps2(selt));
            else
                tmps2(selt) = tmps2(selt) - nanmedian(tmps2(selt));
            end
        end
    end

    % stats
    tmp1 = tmps2(igrp==1);
    tmp2 = tmps2(igrp==2);
    
    if strcmp(avgtype,'mean')
        mu = grpstats(tmps2, igrp, {@mean});
        se = grpstats(tmps2, igrp,{@(x) nanstd(bootstrp(200,@(x) mean(x),x))});

        [~,p,~,s] = ttest2(tmp1,tmp2);
        stat = s.tstat;

    else
        mu = grpstats(tmps2, igrp, {@median});
        se = grpstats(tmps2, igrp,{@(x) nanstd(bootstrp(200,@(x) median(x),x))});

        [p,~,s] = ranksum(tmp1,tmp2);
        stat = s.zval;
    end
    
    % plot
    subplot(nr,nc,2)
    hb = barwitherr(se,mu);

    % finish
    str = sprintf('AMI similarity, wthin vs between monk\nF=%.3g, p=%.3g',...
        stat,p);
    title(str)
    ylabel([avgtype ' AMI'])

    axis square
    set(hb,'facecolor',ones(1,3)*0.8,'BarWidth',0.6)
    set(gca,'xticklabel',{'within','between'},'xlim',[0.3 2+0.7])
    hax(2) = gca;

        
   % ===================================================
    % per task
    grp = {{'1-1'},{'2-2'}};
    selg = contains(itask2,[grp{:}]);
    
    tmps2 = tmps(selg);
    igrp = contains(itask2(selg),grp{2})+1;
    imonk_tmp = imonk2(selg);
    
    if removeMeans
        for it=1:numel(umonk2)
            selt = contains(imonk_tmp,umonk2{it});
            if strcmp(avgtype,'mean')
                tmps2(selt) = tmps2(selt) - nanmean(tmps2(selt));
            else
                tmps2(selt) = tmps2(selt) - nanmedian(tmps2(selt));
            end
        end
    end

    % stats
    tmp1 = tmps2(igrp==1);
    tmp2 = tmps2(igrp==2);
    
    if strcmp(avgtype,'mean')
        mu = grpstats(tmps2, igrp, {@mean});
        se = grpstats(tmps2, igrp,{@(x) nanstd(bootstrp(200,@(x) mean(x),x))});

        [~,p,~,s] = ttest2(tmp1,tmp2);
        stat = s.tstat;

    else
        mu = grpstats(tmps2, igrp, {@median});
        se = grpstats(tmps2, igrp,{@(x) nanstd(bootstrp(200,@(x) median(x),x))});

        [p,~,s] = ranksum(tmp1,tmp2);
        stat = s.zval;
    end
    
    % plot
    subplot(nr,nc,3)
    hb = barwitherr(se,mu);

    % finish
    str = sprintf('AMI similarity, task/task vs notask/notask monk\nF=%.3g, p=%.3g',...
        stat,p);
    title(str)
    ylabel([avgtype ' AMI'])

    axis square
    set(hb,'facecolor',ones(1,3)*0.8,'BarWidth',0.6)
    set(gca,'xticklabel',utask,'xlim',[0.3 2+0.7])
    hax(3) = gca;
    
    
    % save
    if saveFig
        sname = [figdir '/stability_datasets_final'];
        save2pdf(sname)
    end
    foo=1;
end

%% stability of modules between datasets
if plotDatasetStability
    ilag = 1;
    avgtype = 'mean';
    removeMeans = 1;
    subtractRand = 0;

    
    if calcStabilityDatasets
        nrand_s = 100;

        ndat = max(idat);

        Sh = nan(ndat,ndat);
        Shr = nan(ndat,ndat,nrand_s);
        irep=0;
        nsmp = numel(Sh);
        for id1=1:ndat
            for id2=1:ndat
                if id1==id2; continue; end
                irep=irep+1;

                q1 = squeeze(Qlab(ilag,id1,:))+1;
                q2 = squeeze(Qlab(ilag,id2,:))+1;

                % stability
                bad = isnan(q1) | isnan(q2);
                mx = ami(q1(~bad),q2(~bad));

                if isnan(mx)
                    foo=1;
                end
                % store
                Sh(id1,id2) = mx;

                % random
                good = find(~bad);
                qr2 = nan(size(q2));
                for ir=1:nrand_s
                    idx = good( randperm(numel(good)) );
                    qr2(good) = q2(idx);
                    mx = ami(q1(~bad),qr2(~bad));
                    Shr(id1,id2,ir) = mx;
                end
            end
        end
        
        if subtractRand
            Sh = Sh - nanmean(Shr,3);
        end
    end

    % prep
    [umonk,~,imonk] = unique(taskInfo.ck);
    umonk = cellfun(@(x) x(1:2),umonk,'un',0);
    if 0
        [utask,~,itask] = unique(taskInfo.Condition);
    else
        utask = {'noTask','task'};
        itask = contains(taskInfo.Condition,'feeder')+1;
    end
    %[ugrp,~,igrp] = unique([imonk,itask],'rows');

    % prep factors
    tmps = [];
    tmpsr = [];
    imonk2 = {};
    itask2 = {};

    cmb = nchoosek(1:size(taskInfo,1),2);
    for ic=1:size(cmb,1)
        ii1 = cmb(ic,1);
        ii2 = cmb(ic,2);
        tmps(ic) = Sh(ii1,ii2);
        tmpsr(ic,:) = Shr(ii1,ii2,:);
        imonk2{ic} = sprintf('%g-%g',imonk(ii1),imonk(ii2));
        itask2{ic} = sprintf('%g-%g',itask(ii1),itask(ii2));
    end
    imonk2(strcmp(imonk2,'2-1')) = {'1-2'};
    itask2(strcmp(itask2,'2-1')) = {'1-2'};

    % stats on similarity by subject and task
    G = {imonk2, itask2};
    if strcmp(avgtype,'mean')
        [P,T,STATS,TERMS]=anovan(tmps,G,'varnames',{'monk','task'},'display','off');
    else
        itmp = tiedrank(tmps);
        [P,T,STATS,TERMS]=anovan(itmp,G,'varnames',{'monk','task'},'display','off');
    end
    
    if saveFig
        sname = [figdir '/stability_anova_taskSubject.xlsx'];
        writecell(T,sname) % better to use full path
    end

    % stats on within vs between subejct/task
    imonk_w = contains(imonk2,'1-1')*1;
    itask_w = contains(itask2,'1-1')*1;

    G = {imonk_w, itask_w};
    [P2,T2,STATS2,TERMS2]=anovan(tmps,G,'varnames',{'withinMonk','withinTask'},'display','off');

    if saveFig
        sname = [figdir '/stability_anova_taskSubject_withinBetween.xlsx'];
        writecell(T2,sname) % better to use full path
    end

    % overall structure?
    mu = nanmean(Sh(:));
    mur = squeeze(nanmean(nanmean(Shr),2));
    p = sum(mur > mu) ./ nrand_s;

    % get similarity between monkeys, tasks
    figure
    nr = 1; nc = 3;
    set_bigfig(gcf,[0.8 0.4])

    % ===================================================
    % for all conditons
    MU = []; SE = [];
    MUr = []; SEr = [];
    [umonk3,~,imonk3] = unique(imonk2);
    [utask3,~,itask3] = unique(itask2);
    for im=1:numel(umonk3)
        for it=1:numel(utask3)
            sel = imonk3==im & itask3==it;

            % store
            tmp = tmps(sel);
            [MU(im,it),SE(im,it)] = avganderror(tmp(:),avgtype);

            for ir=1:nrand_s
                tmpr = tmpsr(sel,:);
                [MUr(im,it,ir),SEr(im,it,ir)] = avganderror(tmpr(:),avgtype);
            end
        end
    end

    pm = sum(MUr > MU,3) ./ nrand_s;
    pm = bonf_holm(pm);

    subplot(nr,nc,1)
    xx = 1:numel(utask3);
    yy = 1:numel(umonk3);
    imagesc(xx,yy,MU)

    hold all
    [ir,ic] = find(pm<0.05);
    plot(ic,ir,'w.','markersize',10)

    % finish
    str = sprintf('AMI similarity\nAnovan, monk: F=%.3g, p=%.3g\ntask: F=%.3g, p=%.3g',...
        T{2,6},T{2,7},T{3,6},T{3,7});
    title(str)

    xlab = cellfun(@(x) [utask{str2num(x(1))} '-' utask{str2num(x(3))}],utask3,'un',0);
    ylab = cellfun(@(x) [umonk{str2num(x(1))} '-' umonk{str2num(x(3))}],umonk3,'un',0);

    axis square
    set(gca,'xtick',xx,'xticklabel',xlab,'ytick',yy,'yticklabel',ylab)
    colorbar
    %set(gca,'clim',[0 max(get(gca,'clim'))])
    colormap(copper)

    % ===================================================
    % per subject/task
    
    hax = [];
    for ig=1:2
        if ig==1 % monkey
            G = imonk;
            G3 = imonk3;
            u3 = umonk3;
            gstr = umonk;
            lab = ylab;
            tstr = 'subject';
            
            tmps2 = tmps;
            if removeMeans
                for it=1:numel(utask3)
                    sel = itask3==it;
                    if strcmp(avgtype,'mean')
                        tmps2(sel) = tmps(sel) - nanmean(tmps(sel));
                    else
                        tmps2(sel) = tmps(sel) - nanmedian(tmps(sel));
                    end
                end
            end
        else % task
            G = itask;
            G3 = itask3;
            u3 = utask3;
            gstr = utask;
            lab = xlab;
            tstr = 'task';
                            
            tmps2 = tmps;
            if strcmp(avgtype,'mean')
                tmps2(sel) = tmps(sel) - nanmean(tmps(sel));
            else
                tmps2(sel) = tmps(sel) - nanmedian(tmps(sel));
            end
        end
        uG = unique(G);

        % anova stats on between/within
        g1 = cellfun(@(x) u3{x}(1),num2cell(G3),'un',0);
        g2 = cellfun(@(x) u3{x}(3),num2cell(G3),'un',0);
        selw = strcmp(g1,g2)+1;
        [p_wb,t_wb,s_wb]=anova1(tmps2,selw,'off');

        % stats on the cells
        
        if strcmp(avgtype,'mean')
            [p_c,t_c,s_c]=anovan(tmps2,G3,'display','off');
        else
            itmp = tiedrank(tmps2);
            [p_c,t_c,s_c]=anovan(itmp,G3,'display','off');
        end
        if 0
            [c,m,h,nms] = multcompare(s_c,'display','off');
        else
            cmb = combnk(1:numel(u3),2);
            tmpp=[];
            for ic=1:size(cmb,1)
                sel1 = G3==cmb(ic,1);
                sel2 = G3==cmb(ic,2);

                if strcmp(avgtype,'mean')
                    [~,p,~,s] = ttest2(tmps2(sel1),tmps2(sel2));
                    s = s.tstat;
                else
                    [p,~,s] = ranksum(tmps2(sel1),tmps2(sel2));
                    s = s.zval;
                end
                tmpp(ic,:) = [s,p];
            end
            tmpp(:,2) = bonf_holm(tmpp(:,2));
            c = [cmb, nan(size(cmb,1),2),tmpp];
        end

        if saveFig
            sname = [figdir '/stability_anova_' tstr '_withinBetween.xlsx'];
            writecell(t_wb,sname) % better to use full path

            tmp = [t_c; cell(1,size(t_c,2)); [num2cell(c),cell(size(c,1),1)]];
            sname = [figdir '/stability_anova_' tstr '_posthoc.xlsx'];
            writecell(tmp,sname) % better to use full path
        end

        % get means
        MU = []; SE = [];
        for im=1:numel(u3)
            sel = G3==im;
            tmp = tmps2(sel);
            [MU(im),SE(im)] = avganderror(tmp,avgtype);
        end

        % resort for viz
        is = [1 3 2];

        subplot(nr,nc,ig+1)
        hb = barwitherr(SE(is),MU(is));

        % finish
        str = sprintf('AMI similarity for %s\nAnovan(within/bet), F=%.3g, p=%.3g\nsortIndex=%s',...
            tstr,t_wb{2,5},t_wb{2,6},mat2str(is));
        title(str)
        ylabel('Mean AMI')

        axis square
        set(hb,'facecolor',ones(1,3)*0.8,'BarWidth',0.6)
        set(gca,'xticklabel',lab(is),'xlim',[0.3 numel(u3)+0.7])
        hax(ig) = gca;
    end
    %setaxesparameter(hax,'ylim')

    % save
    if saveFig
        sname = [figdir '/stability_datasets'];
        save2pdf(sname)
    end

    foo=1;
end




%% example transition matrices and modules
if plotExampleModules
    %theseLags = [1 10 100 1000];
    theseLags = [1];

    if plotExamplesRandom
        ir = randi(nrand);
        PO = PO_in_rand(:,:,:,:,ir);
        QQ = Qr(:,:,ir);
        tmpdir = [figdir '/_example_modularity_random'];
    else
        PO = PO_in;
        %QQ = Q;
        Qt = Qh_mx;
        Qc = squeeze(Qh(ilag,:,:))';
        tmpdir = [figdir '/_example_modularity'];
    end
    if ~exist(tmpdir); mkdir(tmpdir); end

    % loop over all datasets
    fprintf('plotting transition matrices: ')
    for id=1:max(idat)
        fprintf('%g,',id)
        
        figure('visible','off','name',sprintf('ex modularity, id=%g',id));
        nr = numel(theseLags); nc = 4;
        set_bigfig(gcf,[0.35 min(1,0.15*numel(theseLags))])


        for ii=1:numel(theseLags)
            ilag = lags == theseLags(ii);
            p = PO(:,:,ilag,id);
            lab = squeeze(Qlab(ilag,id,:))+1;

            % plot orig
            ns = 1 + nc*(ii-1);
            subplot(nr,nc,ns)
            out1 = plot_sorted_transition(p,lab,0);
            title('orig')
            
            % plot sorted
            ns = 2 + nc*(ii-1);
            subplot(nr,nc,ns)
            out2 = plot_sorted_transition(p,lab);
            title('sorted')
        end

        % plot modularity by cut
        q = Qc(:,id);
        [mx,imx] = max(q);
        
        ns = 3;
        subplot(nr,nc,ns)
        plot(Cutoffs,q,'r.-')
        hold all
        plot(Cutoffs(imx),q(imx),'ko')

        title('modularity by cut at lag=1');
        xlabel('nclust')
        ylabel('modularity')
        axis square

        
        % plot modularity by transition lag
        q = Qt(:,id);
        ilag = ismember(lags,theseLags);
        ql = q(ilag);

        ns = 4;
        subplot(nr,nc,ns)
        plot(lags,q,'r.-')
        hold all
        plot(theseLags,ql,'ko')

        title('modularity by transition lag')
        xlabel('lag')
        ylabel('modularity')
        axis square
        set(gca,'xscale','log')

        

        if saveFig
            sname = sprintf('%s/example_modularity_id%g',tmpdir,id);
            save2pdf(sname,gcf)
            close(gcf)
        end
    end
    fprintf('\n')  
end



%% different figure
if plotExampleTransition2Modularity
    isVis = 'off';
    
    ilag = 1;
    for id=1:max(idat)
        %id = 13;

        PO = PO_in;
        QQ = Q;
        tmpdir = [figdir '/_example_modularity_lag1'];    % plot original transition matrix
        if ~exist(tmpdir); mkdir(tmpdir); end


        p = PO(:,:,ilag,id);
        lab = squeeze(Qlab(ilag,id,:))+1;
        qh = squeeze(Qh(ilag,id,:));
        T = tmph{ilag,id}(:,1:3);
        T(:,1:2) = T(:,1:2)+1;

        xx = 1:nstate;

        % start figure
        figure('visible',isVis)
        
        nr = 1; nc = 3;
        set_bigfig(gcf,[0.7 0.4])

        % plot unsorted matrix
        subplot(nr,nc,1)
        imagesc(xx,xx,p)

        xlabel('Pose')
        ylabel('Pose')
        colorbar
        axis square

        % plot sorted matrix
        ns = 2;
        subplot(nr,nc,ns)
        out = plot_sorted_transition(p,lab);

        xlabel('Sorted Pose')
        ylabel('Sorted Pose')
        colorbar
        axis square
        hold all

        ax2 = gca;

        % plot modularity score
        %xs = 2:nstate;
        xs = Cutoffs;
        [mx,imx] = max(qh);

        subplot(nr,nc,3)
        plot(xs,qh,'k.-','linewidth',2)
        hold all
        plot(xs(imx),mx,'r^','markerfacecolor','r')
        text(xs(imx),mx*1.1,num2str(xs(imx)))

        axis square
        xlabel('Threshold')
        ylabel('modularity score')
        
        pos1 = get(ax2,'position');
        pos2 = get(gca,'position');
        pos2 = [pos2(1:2) pos1(3:4)];
        set(gca,'position',pos2)

        % plot graph
        %{
        tmpp = zeros(size(p2));
        for ii=1:nstate
            for jj=1:nstate
                if lab(ii)==lab(jj)
                    tmpp(ii,jj) = 1;
                end
            end
        end
        %tmpp = p2;
        tmpp(tmpp<0.1) = 0;
        g = digraph(tmpp);

        subplot(nr,nc,4)
        plot(g,'layout','subspace')
        %}
        % save
        if saveFig
            sname = sprintf('%s/data%g',tmpdir,id);
            save2pdf(sname)

        end
    end
end



%% video of modules
if plotModuleVids
    figdir2 = [figdir '/vid_module3'];
    if ~exist(figdir2); mkdir(figdir2); end

    % convert pose cluster to module cluster
    nplot = 10;
    %timeLim = [1 7]*fs;
    timeLim = [0.1 7]*fs;

    ilag = 1;

    C_mod = nan(size(idat));
    for id=12:max(idat)%17%[13 17]%1:max(idat) %12:18
        fprintf('==== data %g ====\n',id)
        sel = idat==id;
        tmpc = C(sel);

        % module
        oldval = 1:nstate;
        newval = squeeze(Qlab(ilag,id,:))+1;
        nPerModule = accumarray(newval(~isnan(newval)),1,[max(newval) 1]);
        
        tmpc = changem(tmpc,newval,oldval);
        C_mod(sel) = tmpc;

        % make example videos of each module
        for ic=1:max(newval) %[1:3 5:max(newval)]
            %if id==1 && ic==1; continue; end
            sel2 = sel & C_mod==ic;
            [st,fn] = find_borders(sel2);

            % median times
            if 1
                d = fn-st+1;
                mu = median(d);
                [ds,is] = sort(d);
                imu = nearest(ds,mu);

                good = is( imu-nplot/2:imu+nplot/2 );
            else
                good = find(d >= timeLim(1) & d <= timeLim(2));
            end

            %mn = min(numel(good),nplot);
            if 0
                flag=1;
                im=0;
                iplot=0;
                while flag
                    im=im+1;
                    im2 = good(im);
                    smps = [st(im2) fn(im2)];
                    if im==numel(good) || iplot==nplot
                        flag=0;
                    end
                    n = numel(unique(C(smps(1):smps(2))));
                    if n<=1
                        continue
                    else
                        iplot=iplot+1;
                    end

                    % paths
                    savefold = sprintf('%s/data%g_seq%g',figdir2,id,ic);
                    if ~exist(savefold); mkdir(savefold); end

                    tic
                    pose_vid2(X_notran,labels,C,pose_mu,com,savefold,smps,ic,smps(1))
                    toc
                end
            else
                for im=1:numel(good)
                    im2 = good(im);
                    smps = [st(im2) fn(im2)];

                    % paths
                    savefold = sprintf('%s/data%g_seq%g',figdir2,id,ic);
                    if ~exist(savefold); mkdir(savefold); end

                    tic
                    pose_vid2(X_notran,labels,C,pose_mu,com,savefold,smps,ic,smps(1))
                    toc
                end
                
            end
        end
    end

    foo=1;
end






%% ========================================================================
% ========================================================================
% MISC

function call_cluster(PO,mpath,dataname)
    [~,~,pyenvpath,~] = set_pose_paths(0);

    % call python
    verbose = 1;
    cmd = 'fit_louvain';
    savepath = mpath;

    % send to python
    name_in = [dataname '_in.mat'];

    dat = [];
    dat.po = PO;
    save(name_in,'-struct','dat')
    %save(name_in,'-struct','dat','-v7.3')

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

end


function call_cluster_hier(PO,mpath,dataname)
    [~,~,pyenvpath,~] = set_pose_paths(0);

    % call python
    verbose = 1;
    cmd = 'fit_paris';
    savepath = mpath;

    % send to python
    name_in = [dataname '_in.mat'];

    dat = [];
    dat.po = PO;
    save(name_in,'-struct','dat')
    %save(name_in,'-struct','dat','-v7.3')

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

end
