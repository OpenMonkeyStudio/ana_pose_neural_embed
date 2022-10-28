function out = ana_depth_corr(cfg,Var,SDF,iarea)

% checks
cfg = checkfield(cfg,'figdir','needit');

cfg = checkfield(cfg,'varname','needit');
cfg = checkfield(cfg,'monk','needit');

cfg = checkfield(cfg,'model_binned',0);
cfg = checkfield(cfg,'nbin',10);
cfg = checkfield(cfg,'sort_var',0);

% extract
figdir = cfg.figdir;

modelGridPoints = cfg.model_binned;
varname = cfg.varname;
monk = cfg.monk;

% stuff
[parentdir,jsondir,pyenvpath,rpath,binpath,codepath,cagepath] = set_pose_paths(0);



%% get depths
fprintf('importing depth info... \n')

if strcmp(monk,'yo')
    fullname = 'yoda';
else
    fullname = 'woodstock';
end

% import depths
depthname = [cagepath '/drive_map_andInfo_' fullname '.xlsx'];
tbl_turns = readtable(depthname,'Sheet','TurnCountRecord');

Depth = [];
days = {SDF.day};
for id=1:numel(days)
    thisDate = days{id};
    depth = get_depth(thisDate,tbl_turns);
    Depth(:,id) = depth;
end

% get the channel locations
chan_loc = readtable(depthname,'Sheet','DriveMap');
chan_loc = table2array(chan_loc);

% prep
chan = [SDF.ch];

depth2 = [];
for id=1:numel(SDF)
    ich = SDF(id).ch;
    depth2(id,1) = Depth(ich,id);
end
depth2(depth2==0) = nan;

[xc,yc] = find(~isnan(chan_loc));
ichan = chan_loc(sub2ind(size(chan_loc),xc,yc));
[~,is] = sort(ichan);
xc = xc(is);
yc = yc(is);

xc2 = xc(chan);
yc2 = yc(chan);

% sort values? for rubustness
if cfg.sort_var
    [~,Var] = sort(Var);
end

%% plot 3D means
pos_lab = {'ant-post','lat-med','depth'};
nbin = ones(1,3)*cfg.nbin;

xedge=linspace(min(xc2),max(xc2)*1.01,nbin(1)+1);
yedge=linspace(min(yc2),max(yc2)*1.01,nbin(1)+1);
zedge=linspace(min(depth2),max(depth2)*1.01,nbin(1)+1);

[n,xedge,yedge,zedge,binx,biny,binz] = hist_3d(xc2,yc2,depth2,xedge,yedge,zedge);
xe = mean([xedge(1:end-1); xedge(2:end)]);
ye = mean([yedge(1:end-1); yedge(2:end)]);
ze = mean([zedge(1:end-1); zedge(2:end)]);

% prep
sel = n~=0; %>0.005;
n2 = n(sel);

[ix,iy,iz] = ind2sub(size(n),find(sel));
x = xe(ix)';
y = ye(iy)';
z = ze(iz)';

bad = isnan(depth2);
k = accumarray([binx(~bad),biny(~bad),binz(~bad)],Var(~bad),size(n));
k = k ./ n;
k = k(sel);
[nk,kedge,ik] = histcounts(k,20);
ke = mean([kedge(1:end-1); kedge(2:end)]);

% fit model
iarea2 = [];
for ii=1:size(n,1)
    for jj=1:size(n,1)
        for kk=1:size(n,1)  
            selb = binx==ii & biny==jj & binz==kk;
            a = iarea(selb);
            iarea2(ii,jj,kk) = mode(a);
        end
    end
end
iarea2 = iarea2(sel); 


if modelGridPoints
    X = [x,y,z];
    Y = k;
else
    X = [xc2,yc2,depth2];
    Y = Var;
end
X(bad,:) = [];
Y(bad) = [];
X = zscore(X);

vars = [pos_lab];
mdl = fitlm(X,Y,'varnames',[vars,{varname}]);


[P,F,R] = coefTest(mdl);
pcoef = mdl.Coefficients.pValue;
coef = mdl.Coefficients.Estimate;

%% PLOT
% plot
s = sprintf('depth vs %s corr, model-grid=%g, nbin=%g',varname,modelGridPoints,cfg.nbin);
figure('name',s)
nr=4; nc = 4;
set_bigfig(gcf,[0.3 0.9])

% --------------------------------------------------------
% bubbleplot
ns=1:(nr-1)*nc;
subplot(nr,nc,ns)
if 0
    sz = 20;
else
    sz = n2;
end
[lh,th,colmap] = bubbleplot(x,y,z,sz,ik,'o','markerSizeLimits',[10 40]);

axis square
set(gca,'zdir','reverse')
view(-40,30)

colormap(colmap)
hc=colorbar('location','southoutside');
%pos = get(hc,'position');
%pos2 =[0.93,pos(2) + 0.2,pos(3),pos(4)/2];
%pos2 =[pos(1),pos(2) + 0.2,pos(3),pos(4)/2];
%set(hc,'position',pos2);

uk = unique(ik);
ticks = get(hc,'ticks');
ticks = linspace(ticks(1),ticks(end),numel(uk));
ticks = ticks+mean(diff(ticks))/2;
tlabel = cellfun(@(x) num2str(x,'%.2g'),num2cell(ke(uk)),'un',0);
set(hc,'ticks',ticks,'ticklabels',tlabel);

str = sprintf('%s per location, model binned val=%g,\nmodel F=%.3g, p=%.3g, R=%.3g\n%s B=%.3g, p=%.3g\n%s B=%.3g, p=%.3g\n%s B=%.3g, p=%.3g\n',...
                varname,modelGridPoints,F,P,mdl.Rsquared.Ordinary,pos_lab{1},coef(2),pcoef(2),pos_lab{2},coef(3),pcoef(3),pos_lab{3},coef(4),pcoef(4));
title(str)
% xlabel('anterior-posterior')
% ylabel('lateral-medial')
zlabel('depth (mm)')
set(gca,'xtick',[xe(1) xe(end)],'xticklabel',{'anterior','posterior'})
set(gca,'ytick',[ye(1) ye(end)],'yticklabel',{'lateral','medial'})

% --------------------------------------------------------
% K histogram
ns=(nr-1)*nc+1;
subplot(nr,nc,ns)
plot(ke,nk./sum(nk),'k.-','linewidth',2)

xlabel(varname)
ylabel('prop')

axis square

% --------------------------------------------------------
% anterior posterior
[mu,se] = avganderror_group(binx,Var);

ns=ns+1;
subplot(nr,nc,ns)
shadedErrorBar(xe,mu,se)
xlabel('anterior > posterior')
ylabel(['mean ' varname])

set(gca,'xlim',[xe(1) xe(end)])
axis square


% --------------------------------------------------------
% medial lateral
[mu,se] = avganderror_group(biny,Var);

ns=ns+1;
subplot(nr,nc,ns)
shadedErrorBar(ye,mu,se)
xlabel('medial > lateral')
ylabel(['mean ' varname])

set(gca,'xlim',[ye(1) ye(end)])
axis square

% --------------------------------------------------------
% anterior posterior
[mu,se] = avganderror_group(binz(~bad),Var(~bad));

ns=ns+1;
subplot(nr,nc,ns)
shadedErrorBar(ze,mu,se)
xlabel('shallow > deep')
ylabel(['mean ' varname])

set(gca,'xlim',[ze(1) ze(end)])
axis square

% save
sname = sprintf('%s/depth_corr_%s_modelGrid%g_nbin%g',figdir,varname,modelGridPoints,cfg.nbin);
save2pdf(sname,gcf)

%% output
out = [];
out.mdl = mdl;
out.P = P;
out.F = F;
out.R = R;
out.pcoef = pcoef;
out.coef = coef;

