function status = stack_videos(vids,dst)
% status = stack_videos(vids,dst)

% https://ottverse.com/stack-videos-horizontally-vertically-grid-with-ffmpeg/

verbose = 0;

%vids = {'vid_18260994_full.mp4','vid_18261112_full.mp4'};


% prep command
str = cellfun(@(x) ['-i ' x ' '],vids,'un',0);
str = [str{:}];
str(end) = [];

cmd = sprintf('ffmpeg %s -filter_complex hstack=inputs=%g -preset veryfast %s',str,numel(vids),dst);

% run
if verbose
    [status,result] = system(cmd,'-echo');
else
    [status,result] = system(cmd);
end

