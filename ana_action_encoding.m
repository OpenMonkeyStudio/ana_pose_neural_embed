function out = ana_action_encoding(cfg,SDF,res_mod,C,iarea,idat)
% out = ana_action_encoding(cfg,SDF,res_mod,C,iarea)

% checks
cfg = checkfield(cfg,'sdfpath','needit');
cfg = checkfield(cfg,'figdir','needit');
cfg = checkfield(cfg,'datasets','needit');
cfg = checkfield(cfg,'nstate',max(C));
cfg = checkfield(cfg,'nparallel',15);
cfg = checkfield(cfg,'uarea','needit');

cfg = checkfield(cfg,'get_encoding',1);
    cfg = checkfield(cfg,'testtype','kw');
    cfg = checkfield(cfg,'nrand',20);
    cfg = checkfield(cfg,'nboot',1);
    cfg = checkfield(cfg,'eng_lim',[3, ceil(1*cfg.fs_frame)]);
    cfg = checkfield(cfg,'ilag',1);
    cfg = checkfield(cfg,'theseCuts',[2:8 10:2:20 23:3:31, cfg.nstate]);

cfg = checkfield(cfg,'avgtype','mean');


% extract
datasets = cfg.datasets;
figdir = cfg.figdir;
sdfpath = cfg.sdfpath;
nstate = cfg.nstate;
uarea = cfg.uarea;

testtype = cfg.testtype;
nrand = cfg.nrand;
nboot = cfg.nboot;
eng_lim = cfg.eng_lim;
ilag = cfg.ilag;
theseCuts = cfg.theseCuts;

avgtype = cfg.avgtype;

Qh_lab_all = res_mod.obs.labels_cut;
Cutoffs = res_mod.Cutoffs;

% paths
spkparentpath = fileparts(fileparts(sdfpath));


%% classificaiton on individual neurons
sname = [sdfpath '/' testtype '_actions.mat'];
if cfg.get_encoding
    ncuts = numel(theseCuts);
    nrand2 = nrand+1;
    ndat = numel(SDF);
    
    % open parpool
    if cfg.nparallel > 1 && isempty(gcp('nocreate'))
        myPool = parpool('local',cfg.nparallel);
    end
    
    % loop over cells
    tic

    % prep to prevent broadcast
    C_noBroadcast = {};
    for id=1:numel(datasets)
        C_noBroadcast{id} = C(idat==id);
    end

    F = nan(ndat,ncuts,nboot);
    Fr = nan(ndat,ncuts,nboot,nrand);
    P = nan(ndat,ncuts,nboot);
      
    fprintf('indiv neuron encoding: ')
    parfor id=1:ndat
    %for id=1:ndat
    try
        fprintf('%g,',id)
        
        sdf = SDF(id);

        % get data
        X = sdf.sdf;
        id2 = sdf.id;        

        % prep cuts
        tmp = squeeze(Qh_lab_all(ilag,id2,:,:));
        m = max(tmp,[],2);
        maxCuts = max(m);

        ftmp = [];
        frtmp = [];
        ptmp = [];
        
        dat_cuts = cell(ncuts,3);
        for ic1=1:ncuts
            cut = theseCuts(ic1);
            if cut==nstate
                c = C_noBroadcast{id2};
            else
                if cut > maxCuts
                    c = [];
                    continue
                else
                    % remap
                    ic = ismember(Cutoffs,cut);

                    newval = squeeze(Qh_lab_all(ilag,id2,ic,:))+1;
                    oldval = 1:nstate;
                    c = C_noBroadcast{id2};
                    c = changem(c,newval,oldval);
                    c = clean_states(c,3,0);
                end
            end

            % split into "trials"
            [Xt,y,istate] = collapse_poses(X,c,eng_lim);

            % store
            dat_cuts{ic1,1} = Xt';
            dat_cuts{ic1,2} = y;
            dat_cuts{ic1,3} = c;

        end
        nsmp = cellfun(@numel,dat_cuts(:,2));
        mnsmp = min(nsmp);

        % loopp over all cuts
        for ic1=1:ncuts
            for ib=1:nboot
                Xt = dat_cuts{ic1,1};
                y = dat_cuts{ic1,2};

                if isempty(Xt); continue; end

                if nboot > 1
                    bidx = randperm(numel(Xt),mnsmp);
                    bidx = sort(bidx);
                    Xt = Xt(bidx);
                    y = y(bidx);
                end

                % loop calculations
                for ir=1:nrand2
                    % rand or no?
                    if ir <= nrand
                        ridx = randi( ceil([0.25* 0.75]*numel(y)) );
                        y2 = circshift(y,ridx);
                    else
                        y2 = y;
                    end

                    % run anova
                    if strcmp(testtype,'anova')
                        [~,T,~,~] = anovan(Xt,y2,'display','off');
                        f = T{2,6};
                        p = T{2,7};
                    elseif strcmp(testtype,'kw')
                        [~,T,~] = kruskalwallis(Xt,y2,'off');
                        f = T{2,5};
                        p = T{2,6};
                    else
                        f=nan;
                        p=nan;
                        error('unrecognized test')
                    end

                    % store
                    if ir==nrand2
                        ftmp(ic1,ib) = f;
                        ptmp(ic1,ib) = p;
                        %F(id,ic1,ib) = f;
                        %P(id,ic1,ib) = p;
                    else
                        frtmp(ic1,ib,ir) = f;
                        %Fr(id,ic1,ib,ir) = f;
                    end
                end
            end
        end
        
        F(id,:,:) = ftmp;
        Fr(id,:,:,:) = ftmp;
        P(id,:,:) = ptmp;
    catch err
        fprintf('error on %g\n',id)
        rethrow(err)
    end
    end
    fprintf('\n')

    % save for later
    fprintf('saving...\n')
    save(sname,'F','P','Fr','theseCuts','cfg')

    toc
else
    fprintf('loading...\n')
    load(sname)
end



%% plot encoding

% prep
A = nanmean(F,3);
P2 = P;

% get norms by session
tmpa = nan([size(A) 2]);
tmpa(:,:,1) = A;
for id=1:numel(datasets)
    sel = [SDF.id]==id;

    tmp = A(sel,:);
    %tmp = tmp ./ max(tmp(:));
    tmp = tmp - nanmean(tmp);
    tmpa(sel,:,2) = tmp;
end
A = tmpa;

% get engagement
eng = [];
for id=1:numel(datasets)
    datname = datasets(id).name;
    datfold = [spkparentpath  '/' datname];
    [eng(id),f_eng] = get_task_engagement(datfold,0);
end
eng = eng([SDF.id]);

% finish prep
[G,~,iG] = unique(iarea);

xx = theseCuts(1:end-1);
A_act = A(:,end,:);
A = A(:,1:end-1,:);

% ------------------------------------------------
% plot

figure;
nr = 2; nc = 4;
set_bigfig(gcf,[0.4 0.35],[0.4 0.27])
set(gcf,'units','normalized')
pos = get(gcf,'position');

cols = get_safe_colors(0,[1:5 7]);
tstrs = {'raw F','session-norm F'};

% plot grand mean
[mu,se] = avganderror(A(:,:,1),avgtype,1,1,200);
%[mur,ser] = avganderror(Ar,avgtype);

subplot(nr,nc,1)
h = [];
htmp = shadedErrorBar(xx,mu,se,{'r-'});
h(1) = htmp.mainLine;
hold all

title('encoding strength per session')
xlabel('nclust')
ylabel([avgtype ' F' ])
axis square

for ip=1:2  
    % ---------------------------------------
    % split by area
    [mu,se] = avganderror_group(iG,A(:,:,ip),avgtype,200);

    ns = 2 + nc*(ip-1);
    subplot(nr,nc,ns)
    for ii=1:size(mu,1)
        col = cols(ii,:);
        htmp = shadedErrorBar(xx,mu(ii,:),se(ii,:),{'.-','color',col});
        hold all
        hp(ii) = htmp.mainLine;
    end

    s = sprintf('encoding strength by area\n%s',tstrs{ip});
    title(s)
    xlabel('nclust')
    ylabel([avgtype ' F'])
    if ip==1
        hl = legend(hp,uarea,'location','northwest');
        pos = get(hl,'position');
        pos = [0.05 0.05 pos(3:4)];
        set(hl,'position',pos)
    end
    
    axis square

    % ---------------------------------------
    % plot encoding for lowest level 
    [mu,se] = avganderror_group(iG,A_act(:,:,ip),avgtype,200);
    [~,tmp,~,~] = anovan(A_act(:,:,ip),iG,'display','off');
    p = tmp{2,7};
    t = tmp{2,6};

    ns = 3 + nc*(ip-1);
    subplot(nr,nc,ns)
    for ii=1:size(mu,1)
        m = mu(ii);
        s = se(ii);
        hb = barwitherr(s,ii,m);
        set(hb,'facecolor',cols(ii,:))
        hold all
    end

    s = sprintf('encoding actions\n anovan F=%.3g, p=%.3g',t,p);
    title(s)
    ylabel([avgtype ' F'])

    set(gca,'xtick',1:numel(mu),'xticklabel',uarea)
    axis square

    % ---------------------------------------
    % correlate with engagement
    [r,p,ugroup] = corr_group(iG,eng',A_act,'type','spearman');

    ns = 4 + nc*(ip-1);
    subplot(nr,nc,ns)
    for ii=1:size(mu,1)
        m = r(ii);
        s = nan;
        hb = barwitherr(s,ii,m);
        set(hb,'facecolor',cols(ii,:))
        hold all
    end

    title('correlate with engagment')
    ylabel(['spearman R'])

    set(gca,'xtick',1:numel(mu),'xticklabel',uarea)
    axis square
end

% save
sname = [figdir '/encoding_' testtype '.pdf'];
save2pdf(sname,gcf)

%% pre oputput (for otehr analyses
out = [];
out.A_act = A_act;
out.cfg = cfg;

