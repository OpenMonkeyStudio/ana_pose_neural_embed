function status = write_video_segments(src,dst,segs_in)
% write_video_segments(src,dst,segs)
%
% segs must be in seconds

verbose = 0;

% prepare anti-trimming segs
tmps = sort(segs_in(:));

segs = [];
if tmps(1)>0
    is = 2;
    segs(1,:) = [0 segs_in(1)];
else
    is = 1;
end

for ii=2:2:numel(tmps)-1
    segs(is,:) = [tmps(ii) tmps(ii+1)];
    is=is+1;
end
segs(end+1,:) = [tmps(end) 10*60*60];

% prepare filter strings
segstr = '';
segstr2 = '';
for ii=1:size(segs,1)
    st = round(segs(ii,1));
    fn = round(segs(ii,2));
    s = sprintf('[0:v]trim=start=%d:end=%d,setpts=PTS-STARTPTS[%dv];',...
        st,fn,ii-1);
    segstr = sprintf('%s%s',segstr,s);

    segstr2 = sprintf('%s[%gv]',segstr2,ii-1);
end

segstr2 = sprintf('%sconcat=n=%g:v=1[outv]',segstr2,ii);

% finish command
cmd = sprintf('ffmpeg -i %s -filter_complex "%s%s" -map [outv] %s',...
    src,segstr,segstr2,dst);

if verbose
    [status,result] = system(cmd,'-echo');
else
    [status,result] = system(cmd);
end
