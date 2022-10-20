
% flags
getDepths = 0;

modelGridPoints = 0;

% stuff
monk = datasets(1).name(1:2);

thisVar = 'kwF'; %K,RSAtrans,absRSAtrans

if strcmp(thisVar,'KLdiv')
    Var = K;
elseif strcmp(thisVar,'RSAtrans')
    Var = RSA_trans;
elseif strcmp(thisVar,'absRSAtrans')
    Var = abs(RSA_trans);
elseif strcmp(thisVar,'kwF')
    Var = A_act(:,:,1);
else
    error('which var?')
end

%% get depths
if getDepths
    if strcmp(monk,'yo')
        fullname = 'yoda';
    else
        fullname = 'woodstock';
    end
    
    % import depths
    cagepath = set_ephys_paths(0);
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
end

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

%% plot 3D means
pos_lab = {'ant-post','lat-med','depth'};
nbin = ones(1,3)*10;

xedge=linspace(min(xc2),max(xc2)*1.01,nbin(1)+1);
yedge=linspace(min(yc2),max(yc2)*1.01,nbin(1)+1);
zedge=linspace(min(depth2),max(depth2)*1.01,nbin(1)+1);

if 1
    [n,xedge,yedge,zedge,binx,biny,binz] = hist_3d(xc2,yc2,depth2,xedge,yedge,zedge);
    xe = mean([xedge(1:end-1); xedge(2:end)]);
    ye = mean([yedge(1:end-1); yedge(2:end)]);
    ze = mean([zedge(1:end-1); zedge(2:end)]);
else
    [n edges mid loc] = histcn([xc2,yc2,depth2],xedge,yedge,zedge);
%     n(:,:,end)=[];
%     n(:,end,:) = [];
%     n(end,:,:) = [];
    xe=mid{1};
    ye=mid{2};
    ze=mid{3};
    binx=loc(:,1);
    biny=loc(:,2);
    binz=loc(:,3);
end

% prep
%n = n ./ sum(n(:));
sel = n~=0; %>0.005;
n2 = n(sel);
if 0
%[x,y,z] = meshgrid(xe,ye,ze);
[x,y,z] = ndgrid(xe,ye,ze);
x = x(sel);
y = y(sel);
z = z(sel);
else
    [ix,iy,iz] = ind2sub(size(n),find(sel));
    x = xe(ix)';
    y = ye(iy)';
    z = ze(iz)';
end

% nbin = 20;
% [~,~,ik] = histcounts(K,nbin);
% cols = jet(nbin);
% c = cols(ik,:);

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


if 0
    % compare to OFC
    oldval = 1:numel(uarea);
    newval = [4 1 2 3 5];
    iarea2 = changem(iarea2,newval,oldval);
    uarea2 = uarea(newval);

    % fit
    X = [x,y,z,iarea2];
    vars = [pos_lab,{'area'}];
    % X = [x,y,iarea2];
    % vars = [pos_lab(1:2),{'area'}];
    % Y = k;
    mdl = fitlm(X,Y,'categoricalvars',4,'varnames',[vars,{'KL'}]);
else

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
    mdl = fitlm(X,Y,'varnames',[vars,{thisVar}]);
end
[P,F,R] = coefTest(mdl);
pcoef = mdl.Coefficients.pValue;
coef = mdl.Coefficients.Estimate;

%% 
% plot
figure
nr=4; nc = 4;
set_bigfig(gcf,[0.3 0.6])

% --------------------------------------------------------
% bubbleplot
ns=1:(nr-1)*nc;
subplot(nr,nc,ns)
if 0
    sz = 20;
else
    sz = n2;
end
[lh,th,colmap]=bubbleplot(x,y,z,sz,ik,'o','markerSizeLimits',[10 40]);

axis square
set(gca,'zdir','reverse')
view(-40,30)

colormap(colmap)
hc=colorbar;
pos = get(hc,'position');
pos2 =[0.93,pos(2) + 0.2,pos(3),pos(4)/2];
set(hc,'position',pos2);

uk = unique(ik);
%ticks = 1:numel(uk);
ticks = get(hc,'ticks');
ticks = linspace(ticks(1),ticks(end),numel(uk));
ticks = ticks+mean(diff(ticks))/2;
tlabel = cellfun(@(x) num2str(x,'%.2g'),num2cell(ke(uk)),'un',0);
set(hc,'ticks',ticks,'ticklabels',tlabel);

str = sprintf('%s per location\nmodel F=%.3g, p=%.3g\n%s B=%.3g, p=%.3g\n%s B=%.3g, p=%.3g\n%s B=%.3g, p=%.3g\n',...
                thisVar,F,P,pos_lab{1},coef(2),pcoef(2),pos_lab{2},coef(3),pcoef(3),pos_lab{3},coef(4),pcoef(4));
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

xlabel(thisVar)
ylabel('prop')

axis square

% --------------------------------------------------------
% anterior posterior
[mu,se] = avganderror_group(binx,Var);

ns=ns+1;
subplot(nr,nc,ns)
shadedErrorBar(xe,mu,se)
xlabel('anterior > posterior')
ylabel(['mean ' thisVar])

axis square


% --------------------------------------------------------
% medial lateral
[mu,se] = avganderror_group(biny,Var);

ns=ns+1;
subplot(nr,nc,ns)
shadedErrorBar(ye,mu,se)
xlabel('medial > lateral')
ylabel(['mean ' thisVar])

axis square

% --------------------------------------------------------
% anterior posterior
[mu,se] = avganderror_group(binz(~bad),Var(~bad));

ns=ns+1;
subplot(nr,nc,ns)
shadedErrorBar(ze,mu,se)
xlabel('shallow > deep')
ylabel(['mean ' thisVar])
figdir

axis square

% save
sname = [figdir '/encoding_pos_corr_' thisVar];
save2pdf(sname,gcf)