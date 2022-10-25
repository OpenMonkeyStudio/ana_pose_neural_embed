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
cfg = checkfield(cfg,'figdir','needit');
cfg = checkfield(cfg,'datasets','');
cfg = checkfield(cfg,'fs_frame',30);
cfg = checkfield(cfg,'nparallel',15);
cfg = checkfield(cfg,'uarea','needit');

cfg = checkfield(cfg,'only_nonengage',1);
cfg = checkfield(cfg,'seg_lim',[-1 1]);
cfg = checkfield(cfg,'seg_min',0.2);
cfg = checkfield(cfg,'eng_smooth',1);
    
cfg = checkfield(cfg,'avgtype','median');
cfg = checkfield(cfg,'normtype','');
cfg = checkfield(cfg,'weighted_mean',0);


% extract
datasets = cfg.datasets;
figdir = cfg.figdir;
sdfpath = cfg.sdfpath;
fs_frame = cfg.fs_frame;
uarea = cfg.uarea;

onlyNonEngage = cfg.only_nonengage;
lim = cfg.seg_lim;
minSeg = cfg.seg_min;
smoothWin = ceil(cfg.eng_smooth*fs_frame);

avgtype = cfg.avgtype;
doWeightedMean = cfg.weighted_mean;
normtype = cfg.normtype;

spkparentpath = fileparts(fileparts(sdfpath));

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
        tooShort = d < minSeg*fs_frame;
        istate(tooShort) = [];

        % create indices
        s1 = abs(lim(1)*fs_frame);
        s2 = abs(lim(2)*fs_frame);
        istate(istate <= s1) = [];
        istate(istate + s2 > numel(c)) = [];

        idx = -s1:s2;
        idx = [ repmat(idx,numel(istate),1) + istate ]';
        idx = [idx(:)];

        % extract
        ncell = sum( cellfun(@numel,{sdf.area}) );
        ndat = numel(istate);
        nsmp = diff(lim*fs_frame)+1;

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
        tmp.area = collapse_areas([sdf.area]);

        RES_seg = cat(1,RES_seg,tmp);

        foo=1;
    end
    fprintf('\n')
end

%% PREP
st = lim(1)*fs_frame;
fn = lim(2)*fs_frame;
xtime = [st:fn] ./ fs_frame;
    
% areas
a = cat(2,RES_seg.area)';
[~,iarea] = ismember(a,uarea);

% get mean rate per area
fprintf('extracting seg means per cell...\n')

MU = [];
for id=1:numel(RES_seg)
    tmp = RES_seg(id).seg;
    pre = tmp(:,:,xtime<0);
    post = tmp(:,:,xtime>0);
    c = RES_seg(id).C_seg(:,2); % pre OR post state

    % time series
    if strcmp(normtype,'presegnorm') % z-norm by pre of each seg
        m = nanmedian(pre,3);
        s = mad(pre,1,3);

        if doWeightedMean % weighted mean
            [uc,~,ic] = unique(c);
            tmpm = [];
            for ii=1:numel(uc)
                sel = ic==ii;
                a = nanmedian( (tmp(sel,:,:) - m(sel,:)) ./ s(sel,:),1 );
                tmpm = cat(1,tmpm,a);
            end
            mu = nanmedian(tmpm,1);
            foo=1;
        else
            mu = nanmedian( (tmp-m)./s );        
            %mu = nanmean( (tmp-m)./s );  
        end      
    else
        mu = nanmean(tmp);
    end
    mu = squeeze(mu);

    MU = cat(1,MU,mu);
end
    
% clean
bad = isnan(MU) | abs(MU)>10^10;
MU(bad) = nan;
    

%% PLOTTING

% start figure
figure;
nr = 1; nc = 2;
set_bigfig(gcf,[0.3 0.2])
cols = get_safe_colors(0,[1:5 7]);
    
    
% plot mean timesries, split by area
[mu,se] = avganderror_group(iarea,MU,avgtype,100);
    
p = [];
for ii=1:size(MU,2)
    if strcmp(avgtype,'median')
        p(ii) = kruskalwallis(MU(:,ii),iarea,'off');
        teststr = 'KW';
    else
        p(ii) = anovan(MU(:,ii),iarea,'display','off');
        teststr = 'Anova';
    end
end
p = bonf_holm(p);
    
mup = ones(size(xtime)) * max([mu(:)+se(:)]) * 1.05;
mup(p>0.05) = nan;

subplot(nr,nc,1)
h = [];
for ii=1:size(mu,1)
    if 1
        htmp = shadedErrorBar(xtime,mu(ii,:),se(ii,:),{'-','color',cols(ii,:)},0);
        h(ii) = htmp.mainLine;
    else
        h(ii) = plot(xtime,mu(ii,:),'color',cols(ii,:));
    end
    hold all
end
plot(xtime,mup,'k.')
    
pcl('x',0)
legend(h,uarea,'location','northwest')

title('mean rate, normalized to pre switch')
xlabel('time')
ylabel([avgtype ' baseline-norm rate'])

axis square
    
% plot pre/post
pre = nanmean(MU(:,xtime < 0),2);
post = nanmean(MU(:,xtime > 0),2);
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
strs1 = {'all','nonengage'};
sname = sprintf('%s/sdf_switch_%s_%s_weighted%g.pdf',figdir,strs1{onlyNonEngage+1},normtype,doWeightedMean);
save2pdf(sname)

foo=1;


%% prep output
out = [];
out.RES_seg = RES_seg;
out.cfg = cfg;
out.MU_seg = MU;
out.dSeg = dSeg;

