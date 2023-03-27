function out = ana_switch_sdf(cfg,varargin)
% out = ana_switch_sdf(cfg,SDF,C,frame,iarea,idat)
% out = ana_switch_sdf(cfg,RES_seg)

% prep
if nargin <= 2
    calcSeg = 0;
    RES_seg = varargin{1};
else
    calcSeg = 1;

    SDF = varargin{1};
    C = varargin{2};
    frame = varargin{3};
    iarea = varargin{4};
    idat = varargin{5};
end

% checks
cfg = checkfield(cfg,'sdfpath','');
cfg = checkfield(cfg,'figdir','');
cfg = checkfield(cfg,'plot',1);
cfg = checkfield(cfg,'datasets','');
cfg = checkfield(cfg,'fs_frame',30);
cfg = checkfield(cfg,'nparallel',15);
cfg = checkfield(cfg,'uarea','needit');

cfg = checkfield(cfg,'only_nonengage',1);
cfg = checkfield(cfg,'ana_lim',[-1 1]);
cfg = checkfield(cfg,'plot_lim',[-1 1]);
cfg = checkfield(cfg,'seg_lim',[-1 1]);
cfg = checkfield(cfg,'baseline',[-Inf 0]);
cfg = checkfield(cfg,'seg_min',0.2);
cfg = checkfield(cfg,'eng_smooth',1);
    
cfg = checkfield(cfg,'avgtype','median');
cfg = checkfield(cfg,'normtype','');
cfg = checkfield(cfg,'normavg','');
cfg = checkfield(cfg,'weighted_mean',0);


% extract
datasets = cfg.datasets;
figdir = cfg.figdir;
sdfpath = cfg.sdfpath;
fs_frame = cfg.fs_frame;
uarea = cfg.uarea;

slim = cfg.seg_lim; % segment limit to select
plim = cfg.plot_lim; % limits to plot
lim = cfg.ana_lim; % limits to collapse over
if numel(lim)==2; lim = [lim(1) 0; 0 lim(2)]; end
minSeg = cfg.seg_min;
smoothWin = ceil(cfg.eng_smooth*fs_frame);

onlyNonEngage = cfg.only_nonengage;
avgtype = cfg.avgtype;
doWeightedMean = cfg.weighted_mean;
normtype = cfg.normtype;

spkparentpath = fileparts(fileparts(sdfpath));

if ~exist(figdir) && ~isempty(figdir); mkdir(figdir); end

%% peri-transition PSTHs
%strs1 = {'all','nonengage'};
%sname = sprintf('%s/sdf_switch_%s.mat',sdfpath,strs1{onlyNonEngage+1});
if calcSeg
    RES_seg = [];
    % get all segments
    fprintf('getting segs: ')
    for id=1:numel(datasets)
        name = datasets(id).name;
        fprintf('%g,',id);

        % get SDF
        day = datestr( name(4:end-10) );
        selsdf = ismember({SDF.day},day);
        sdf = SDF(selsdf);

        % concate rates
        Xs = cat(1,sdf.sdf);

        % get state stuff
        sel = idat==id;
        c = C(sel);
        f = frame(sel);

        % state indices
        istate = find( diff(c)~=0 );

        % get rid of too short segs
        d = [diff(istate); numel(c) - istate(end)+1];
        if 1
        tooShort = d < minSeg*fs_frame;
        istate(tooShort) = [];
        d(tooShort) = [];
        end
        
        % get rid of segments where pre/post switch happened within limits
        %s1 = abs(lim(1)*fs_frame);
        %s2 = abs(lim(2)*fs_frame);
        %istate(istate <= s1) = [];
        %istate(istate + s2 > numel(c)) = [];

        % create indices
        s1 = abs(slim(1)*fs_frame);
        s2 = abs(slim(2)*fs_frame);
        bad = (istate <= s1) | (istate + s2 > numel(c)); % get rid of initial/end segments if dont fit in lims
        istate(bad) = []; 
        d(bad) = [];
        
        
        idx = -s1:s2;
        idx = [ repmat(idx,numel(istate),1) + istate ]';
        idx = [idx(:)];

        % extract
        ncell = sum( cellfun(@numel,{sdf.area}) );
        ndat = numel(istate);
        nsmp = diff(slim*fs_frame)+1;

        xs = Xs(:,idx);
        xs = reshape(xs,[ncell,nsmp,ndat]);
        S = permute(xs,[3 1 2]);

        C_seg = [c(istate), c(istate+1)];

        %only non-engaged periods?
        if onlyNonEngage
            % task engagement
            datfold = [spkparentpath  '/' name];
            [eng,f_eng] = get_task_engagement(datfold,smoothWin,f);

            % clean
            idx2 = reshape(idx,nsmp,ndat);
            badidx = false(size(idx2));
            badidx(f(idx2) < f_eng(1)) = 1;
            badidx(f(idx2) > f_eng(end)) = 2;

            tmpidx = find(ismember(f(idx),f_eng),1);
            tmpidx = idx(tmpidx);
            idx2(badidx==1) = tmpidx;
            idx2(badidx==2) = tmpidx;

            % find engaged segments
            tmp = eng(idx2);
            bad = any( tmp > 0 | badidx~=0 );

            % cull
            istate(bad) = [];
            S(bad,:,:) = [];
            C_seg(bad,:) = [];
        end

        % save and store        
        tmp = [];
        tmp.name = name;
        tmp.C_seg = C_seg;
        tmp.seg = S;
        tmp.segt = d;
        tmp.area = collapse_areas([sdf.area]);
        tmp.mu = nanmean(Xs,2);
        tmp.std = nanstd(Xs,[],2);
        tmp.med = nanmedian(Xs,2);
        tmp.mad = mad(Xs,1,2);

        RES_seg = cat(1,RES_seg,tmp);

        foo=1;
    end
    fprintf('\n')
end



%% prep output
out = [];
out.RES_seg = RES_seg;
out.cfg = cfg;

if ~cfg.plot
    return
end 

%% PREP
st = slim(1)*fs_frame;
fn = slim(2)*fs_frame;
xtime = [st:fn] ./ fs_frame;
    
% cull too short segs
% fprintf('culling bad segs...\n')
% for id=1:numel(RES_seg)
%     d = RES_seg(id).segt ./ fs_frame;
%     bad = d < minSeg;
%     RES_seg(id).C_seg(bad,:) = [];
%     RES_seg(id).seg(bad,:) = [];
%     RES_seg(id).segt(bad,:) = [];
% end

% areas
a = cat(2,RES_seg.area)';
[~,iarea] = ismember(a,uarea);

% get mean rate per area
fprintf('extracting seg means per cell...\n')

MU = nan(numel(iarea),numel(xtime));
dR = nan(numel(iarea),1);

fn = 0;
for id=1:numel(RES_seg)
    fprintf('%g,',id)
    tmp = RES_seg(id).seg;

    base = tmp(:,:,xtime >= cfg.baseline(1) & xtime <= cfg.baseline(2));
    pre = tmp(:,:,xtime>=lim(1,1) & xtime<lim(1,2));
    post = tmp(:,:,xtime>lim(2,1) & xtime<=lim(2,2));
    
    c = RES_seg(id).C_seg(:,2); % pre OR post state

    % time series
    if strcmp(normtype,'zindivseg') % z-norm by pre of each seg
        if strcmp(cfg.normavg,'median')
            m = nanmedian(base,3);
            s = mad(base,1,3);
        else
            m = nanmean(base,3);
            s = nanstd(base,1,3);
        end

        if doWeightedMean % weighted mean
            [uc,~,ic] = unique(c);
            tmpm = [];
            for ii=1:numel(uc)
                sel = ic==ii;
                tmp2 = (tmp(sel,:,:) - m(sel,:)) ./ s(sel,:);
                if strcmp(cfg.normavg,'median')
                    a = nanmedian(tmp2,1);
                else
                    a = nanmean(tmp2,1);
                end
                tmpm = cat(1,tmpm,a);
            end
            if strcmp(cfg.normavg,'median')
                mu = nanmedian(tmpm,1);
            else
                mu = nanmean(tmpm,1);
            end
            foo=1;
        else
            mu = (tmp-m)./s;
            if strcmp(cfg.normavg,'median')
                mu = nanmedian(mu);        
            else
                mu = nanmean(mu);  
            end
        end      
    elseif strcmp(normtype,'zavgseg')
        selt = xtime>=base(1,1) & xtime<=base(1,2);

        if strcmp(cfg.normavg,'median')

            mu = nanmedian(tmp);

            p = mu(:,:,selt);
            m = nanmedian(p,3);
            s = mad(p,1,3);

            mu = (mu - m) ./ s;
        else
            mu = nanmean(tmp);

            p = mu(:,:,selt);
            m = nanmean(p,3);
            s = nanstd(p,1,3);

            mu = (mu - m) ./ s;
        end
        
    elseif strcmp(normtype,'zsess')
        if 0
            m = RES_seg(id).mu;
            s = RES_seg(id).std;
        else
            m = RES_seg(id).med;
            s = RES_seg(id).mad;
        end
        
        mu = (tmp - m') ./ s';
        mu = nanmedian(mu);
        
        foo=1;
    elseif strcmp(normtype,'propindivseg')
        mu = tmp ./ sum(tmp,3);
        mu = nanmedian(mu);
        
        foo=1;
    elseif strcmp(normtype,'rngindivseg')
        mu = tmp - min(tmp,[],3);
        mu = mu ./ max(mu,[],3);
        mu = nanmedian(mu);
    else
        if strcmp(cfg.normavg,'median')
            mu = nanmedian(tmp);
        else
            mu = nanmean(tmp);
        end
    end
    mu = squeeze(mu);

    % straight up difference
    d = nanmedian(post,3) - nanmedian(pre,3);
    d = nanmedian(d)';
    
    
    % update
    st = fn+1;
    fn = fn+size(mu,1);
    if numel(fn) > size(MU,1)
        foo=1;
    end
    MU(st:fn,:) = mu;
    %MU = cat(1,MU,mu);
    
    dR(st:fn) = d;
end
fprintf('\n')

% clean
bad = isnan(MU) | abs(MU)>10^10;
MU(bad) = nan;
    

%% PLOTTING
if cfg.plot
    % start figure
    figure;
    nr = 1; nc = 2;
    set_bigfig(gcf,[0.6 0.4])
    cols = get_safe_colors(0,[1:5 7]);


    % plot mean timesries, split by area
    selt = xtime >= plim(1) & xtime <= plim(2);
    tmpmu = MU(:,selt);
    xtime2 = xtime(selt);
    
    [mu,se] = avganderror_group(iarea,tmpmu,avgtype,100);

    p = [];
    for ii=1:size(tmpmu,2)
        if strcmp(avgtype,'median')
            p(ii) = kruskalwallis(tmpmu(:,ii),iarea,'off');
            teststr = 'KW';
        else
            p(ii) = anovan(tmpmu(:,ii),iarea,'display','off');
            teststr = 'Anova';
        end
    end
    p = bonf_holm(p);

    mup = ones(size(xtime2)) * max([mu(:)+se(:)]) * 1.05;
    mup(p>0.05) = nan;

    subplot(nr,nc,1)
    h = [];
    for ii=1:size(mu,1)
        if 1
            htmp = shadedErrorBar(xtime2,mu(ii,:),se(ii,:),{'-','color',cols(ii,:)},0);
            h(ii) = htmp.mainLine;
        else
            h(ii) = plot(xtime2,mu(ii,:),'color',cols(ii,:));
        end
        hold all
    end
    plot(xtime2,mup,'k.')

    pcl('x',0)
    legend(h,uarea,'location','northwest')

    s = sprintf('%s rate, norm=%s, baseline=[%g %g]',avgtype,cfg.normtype,cfg.baseline(1),cfg.baseline(2));
    title(s)
    xlabel('time')
    ylabel([avgtype ' baseline-norm rate'])

    axis square

    % plot pre/post
    pre = nanmean(MU(:,xtime >= lim(1,1) & xtime <= lim(1,2)),2);
    post = nanmean(MU(:,xtime >= lim(2,1) & xtime <= lim(2,2)),2);
    
    dSeg = post - pre;
    [mu,se] = avganderror_group(iarea,dSeg,avgtype,100);
    [~,T] = kruskalwallis(dSeg,iarea,'off');
    fa = T{2,5};
    pa = T{2,6};

    p = [];
    for ia=1:numel(uarea)
        sela = iarea==ia;
        tmp = dSeg(sela);
        p(ia) = signrank(tmp);
    end
    mup = ones(size(mu)) * max(mu+se)*1.05;
    mup(p>0.05) = nan;

    subplot(nr,nc,2)
    hb=[];
    for ia=1:numel(uarea)
        hb(ia) = barwitherr(se(ia),ia,mu(ia));
        set(hb(ia),'facecolor',cols(ia,:));
        hold all
    end
    hold all
    plot(1:numel(uarea),mup,'k.')

    s = sprintf('post-pre diff\n%s X2=%.3g, p=%.3g',teststr,fa,pa);
    title(s)
    ylabel([avgtype ' norm post-pre diff'])

    set(gca,'xtick',1:numel(uarea),'xticklabel',uarea)
    axis square


    % save
    if ~isempty(figdir)
        strs1 = {'all','nonengage'};
        sname = sprintf('%s/sdf_switch_%s_%s_weighted%g.pdf',figdir,strs1{onlyNonEngage+1},normtype,doWeightedMean);
        save2pdf(sname)
    end

    foo=1;
end

%% continue output
% out = [];
% out.RES_seg = RES_seg;
% out.cfg = cfg;
out.MU_seg = MU;
out.dSeg = dSeg;

