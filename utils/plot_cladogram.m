function out = plot_cladogram(tree,poses,stateMap,nClusters,labels,isVis,plotCladoPoses)
% out = plot_cladogram(tree,poses,stateMap,nClusters,labels,isVis,plotCladoPoses)

%disp('cladogram...')
useBranchPval = 0;
% plotCladoPoses = 1;

nstate = numel(stateMap)/2;
cutoff      = median([tree(end-nClusters+1,3) tree(end-nClusters+2, 3)]);
    
% start figure
tmp = {'off','on'};
figure('name','cladogram','visible',tmp{isVis+1})
if plotCladoPoses
    [nr, nc] = subplot_ratio(nstate,3);
    nr = nr+2;
    %nr = 4; nc = ceil(nstate/(nr-1));
    set_bigfig(gcf,[1 0.7])
    ns_dendo = [1 nc*2];
else
    nr = 1; nc = 1;
    ns_dendo = 1;
    set_bigfig(gcf,[0.25 0.2])
end

% plot claudogram
subplot(nr,nc,ns_dendo)

[hden,~,tTree] = dendrogram(tree,nstate,'colorthreshold',cutoff);
hold all
set(gca,'TickLength',[0.01 0.002])
denax = gca;

% figure out leaf colorings and stuff
color_map = {};

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

for it=1:numel(tTree)
    ip = tTree(it);
    sel = stateMap(:,2)==ip;
    ip2 = stateMap(sel,1);

    color_map{it,1} = cols_in(it,:);
    color_map{it,2} = ip;
    color_map{it,3} = ip2;
end

% reorder leaf IDs
axden = denax;

xt = str2num(get(axden,'xticklabel'));
m = stateMap(xt,1);
m = cellfun(@num2str,num2cell(m),'un',0);
set(axden,'xticklabel',m)

tmp = stateMap(xt,3);
theseLeaves = [1; find(diff(tmp)~=0)+1];
hl = legend(hleaf(theseLeaves),cellfun(@num2str,num2cell(tmp(theseLeaves)),'un',0),'location','eastoutside');
hl.Title.String = 'Module';

% plot example poses?
if plotCladoPoses
    % plot
    ax = [];
    hpose = [];
    for it=1:numel(tTree)
        ip = tTree(it);
        sel = stateMap(:,2)==ip;
        ip2 = stateMap(sel,1);
        
        ns = nc*2 + it;
        subplot(nr,nc,ns)

        m = poses(ip2,:);
        m = m + ip;
        tmph = plot_monkey({m,labels},[],1);
        hpose = cat(1,hpose,tmph);

        set(gca,'xtick',[],'ytick',[],'ztick',[])

        s = sprintf('%g',ip2);
        title(s)
        xlabel('')
        ylabel('')
        zlabel('')

        if 1
            box on
            c = cols_in(it,:);
            set(gca,'Xcolor',c,'ycolor',c,'zcolor',c)
        end

        ax(it) = gca;
        foo=1;
    end
else
    ax = [];
    hpose = [];
end

% ouput
out = [];
out.ax_den = denax;
out.ax_poses = ax;
out.hden = hden;
out.hpose = hpose;
out.color_map = color_map;
out.hleaf = hleaf;