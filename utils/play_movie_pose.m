function play_movie_pose(vidnames,DS,labels,F,C2,thisFs,thisData,saveOpts)
% play_movie_pose(vidnames,DS,labels,F,C2,thisFs)
% play_movie_pose(vidnames,DS,labels,F,C2,thisFs,thisData)
% play_movie_pose(vidnames,DS,labels,F,C2,thisFs,thisData,saveOpts)

% vidnames = {
%     '/mnt/scratch/BV_embed/P_neural_embed/test/vids/vid_18260994_full.mp4'
%     '/mnt/scratch/BV_embed/P_neural_embed/test/vids/vid_18261112_full.mp4'
%     };

% prep
tmp = reshape(DS,[size(DS,1),3,size(DS,2)/3]);


mn = min(min(tmp,[],3));
mx = max(max(tmp,[],3));
cageLim = [mn;mx];

% prep
if iscell(DS)
    smp = DS{2};
    DS = DS{1};
end
if ~isstruct(DS)
    DS = mat2oms(DS,labels,[]);
end

if numel(C2)>0
    plotClusterStream = 1;
    [ulabels,~,ilabels] = unique(C2);
    cols = hsv(max(ilabels)+1);
else
    plotClusterStream = 0;
    ulabels = '-';
    cols = 'k';
end
    
% read in videos
V = {};
for im=1:numel(vidnames)
    V{im} = VideoReader(vidnames{im});
end

fs_vid = 1./V{1}.FrameRate;

if nargin < 6 || isempty(thisFs)
    fs = fs_vid;
else
    fs = thisFs;
end

if nargin < 7 || isempty(thisData)
    thisData = {};
    plotData = 0;
else
    plotData = 1;
end

if nargin < 8 || isempty(saveOpts)
    saveVid = 0;
else
    saveVid = 1;
    
    savepath = saveOpts.savepath;
    newVidObj = VideoWriter(savepath);
    newVidObj.FrameRate = 1./fs_vid;
    
    open(newVidObj)
end

nvid = numel(vidnames);

% plot and position axes
figure
nr = 2; nc = nvid;
pos = [0.5 0.3]; %[0 1]; %[0.5 0.3]
set_bigfig(gcf,[0.35,0.4],pos)

% videos
hax = [];
hv = {};
for im=1:numel(V)
    %fr = max((F(1)-2/fs_vid)*fs_vid,0); % make sure it looks back far enough?
    %fr = fs_vid;
    fr = F(1)*fs_vid;
    V{im}.CurrentTime = fr;
    subplot(nr,nc,im)
    vf = readFrame(V{im});
    hv{im} = image(vf);
    hax(im) = gca;
    %h = imshow(vf); hv{im} = get(h);
    set(gca,'xtick',[],'ytick',[])
    
end

% make videos bigger and properly scaled
if 0
pos = get(hax,'position'); pos = cat(1,pos{:});
xoff = (pos(2,2) - (pos(1,3)+pos(1,1))) * 0.48;
pos2 = pos;
pos2 = [pos(:,1)-xoff pos(:,2)-xoff pos(:,3)+xoff pos(:,4)+xoff];
for ii=1:numel(hax); set(hax(ii),'position',pos2(ii,:)); end
end

% pose
subplot(nr,nc,nvid+1)
out = plot_monkey_slim(DS.coords_3d,cageLim,1);
set(out.ax,'xtick',[],'ytick',[],'ztick',[])
hTitle = title(out.ax,'');

pos = out.cage_dim;
pos = [pos(2,1) pos(1,2) pos(2,3)];
ht = text(pos(1),pos(2),pos(3),'k');
pos = get(out.ax,'position');
ha = annotation('rectangle',pos);
            
% cluster stream
if plotClusterStream
    pos = get(out.ax,'position');
    pos2 = [pos(1) pos(2)-0.05 pos(3) 0.03];
    haxt = axes('position',pos2);
    hc = [];
    for ic=1:max(ilabels)
        sel = ilabels==ic;
        xx = find(sel);
        yy = ones(size(xx));
        hc(ic) = plot(haxt,xx,yy,'.','color',cols(ic,:));
        hold all
    end
    set(gca,'xlim',[1 numel(ilabels)])
    hct = plotcueline('x',1);
    tmph = legend(hc,ulabels);
    tmppos = get(tmph,'position');
    set(tmph,'position',[pos(1)-tmppos(3)-0.02, pos(2), tmppos(3:4)])
end

% data stream
if plotData
    datWin = [-10 10];

    %subplot(nr,nc,nvid+2)
    
    hdax = [];
    hd = [];
    hdt = [];
    for ii=1:size(thisData,1)
        pos = [0.57 0.1*ii 0.33 0.11];
        hdax(ii) = axes('position',pos);
        hd(ii) = plot(hdax(ii),thisData{ii,2},'k');
        hdt(ii) = plotcueline(hdax(ii),'x',1);

        ylabel(thisData{ii,1})
        set(gca,'xlim',[1 numel(thisData{ii,2})])
    end
end

%pause(5)

% play everything simultaneously
for ii=2:numel(F)
    tic;
    fr = F(ii);

    % videos
    for im=1:numel(V)
        v = V{im};
        if 1
            % setting the current time is costly
            if ii==1 || (ii>1 && diff(F(ii-1:ii))>1)
                %disp('skip')
                v.CurrentTime = (fr-1)*fs_vid;
            end

            vf = readFrame(v);
        else
            vf = read(v,fr);
        end
        
        hv{im}.CData = vf;
        %set(hax(im),'cdata',vf)
    end
    
    % update pose
    out = plot_monkey_slim(DS.coords_3d,cageLim,ii,out);
    %toc
    
    if ~isempty(C2)
        ht.String = C2(ii);
        ha.Color = cols(ilabels(ii),:);
    end
    %toc
   
    s = sprintf('frame %g, smp %g',fr,ii);
    hTitle.String = s;
    %toc
    
    % update cluster timeseries
    if plotClusterStream
        set(hct,'xdata',[ii ii])
    end
    
    % udpate data time series
    if plotData
        for id=1:size(thisData,1)
            set(hdt(id),'xdata',[ii ii])
        end
        %set(gca,'xlim',datWin+ii)
    end

    % enforce frme rate
    %drawnow
    %hold off
    
    dT = toc;
    d = fs - dT; % processing takes time
    pause(d);
    
    % write to new video?
    if saveVid
        newf = getframe(gcf);
        writeVideo(newVidObj,newf);
    end
end

% finish
if saveVid
    close(newVidObj)
end