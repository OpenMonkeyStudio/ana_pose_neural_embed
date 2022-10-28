function cluster_example_videos(C,frame,cfg)
% cluster_example_videos(C,frame,cfg)

% dstpath = '/mnt/scratch/BV_embed/P_neural_embed/test/vids';
% 
% idat = ones(size(C));
% 
% vidnames = {'vid_18261112_full.mp4','vid_18261030_full.mp4'};
% vidnames = cellfun(@(x) ['/mnt/scratch/BV_embed/P_neural_embed/' name(1:end-9) '/vids/' x],vidnames,'un',0);

% settings
cfg = checkfield(cfg,'dstpath','needit');
cfg = checkfield(cfg,'vidnames','needit');
cfg = checkfield(cfg,'dThresh',5);
cfg = checkfield(cfg,'nrand',10);
cfg = checkfield(cfg,'suffix','');

% stuff
nstate = max(C);
fs = 30;

if ~exist(cfg.dstpath); mkdir(cfg.dstpath); end

if isrow(frame); frame = frame'; end

% loop over each one
for ic=1:nstate
    fprintf('videos: cluster=%g',ic)

    % find all segments
    sel = C==ic;
    [st,fn] = find_borders(sel);
    %st = frame(st);
    %fn = frame(fn);
    
    d = fn - st + 1;
    tooShort = d < cfg.dThresh;
    st(tooShort) = [];
    fn(tooShort) = [];
    d(tooShort) = [];
    segs = [frame(st) frame(fn)] ./ fs;
    segs = round(segs);

    if size(segs,1)==0
        fprintf('... no segments selected\n')
        continue
    end

    % select a few random one
    mx = min(cfg.nrand,numel(st));
    ii = sort( randperm(numel(st),mx) );
    segs = segs(ii,:);

    % make temporary videos of just these segments
    tmpname = tempname;

    tmpdst = {};
    res = [];
    for is=1:numel(cfg.vidnames)
        [fpath,fname,ext] = fileparts(cfg.vidnames{is});
        d = sprintf('%s_%s%s',tmpname,fname,ext);
        s = cfg.vidnames{is};
        res(is) = write_video_segments(s,d,segs);

        tmpdst{is} = d;
    end

    % now combine them into one for easy viewing
    dst = sprintf('%s/cluster%03g_%s%s',cfg.dstpath,ic,cfg.suffix,ext);        
    stack_videos(tmpdst,dst)

    % clean
    for ii=1:numel(tmpdst)
        delete(tmpdst{ii})
    end

    fprintf('... done\n')
end


