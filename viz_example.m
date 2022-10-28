    
thisID = 8;

%% load
if 1
    % original data
    name = datasets(thisID).name;
    sname = [fileparts(anadir) '/Data_proc_13joint/data_ground/' name '_proc.mat'];
    load(sname)
    labels = data_proc.labels;
    
    % videos
    vidnames = {'vid_18261112_full.mp4','vid_18261030_full.mp4'};
    vidnames = cellfun(@(x) [fileparts(anadir) '/' name '/vids/' x],vidnames,'un',0);   
end

% prep
%plottingLims = [13179,13205;13220,13535;13535,14335;14390,14430;14435,14464;14469,14915;14965,15000]';
%plottingStrs = {'jump down','walk','feeder','walk','jump up','sit1'};

lim = [13180 14800];
%lim = [13100 13220];
    
f = data_proc.frame;
selx = f >= lim(1) & f <= lim(2);
selx2 = idat==thisID & (frame >= lim(1) & frame <= lim(2));

cstr = cellfun(@num2str,num2cell(1:max(C)),'un',0);
Fr = data_proc.frame(selx);
DS = data_proc.data(selx,:);
C2 = cstr(C(selx2));

opts = [];
opts.savepath = [figdir '/example_pose_' name '.avi'];
play_movie_pose(vidnames,DS,labels,Fr,C2,[],[],opts)