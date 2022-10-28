

%ffmpeg -ss 00:00:03 -i inputVideo.mp4 -to 00:00:08 -c:v copy -c:a copy

fs = 30;
segs = [1 30;
        1000 1030];
  
segstr = '';
segstr2 = '';
for ii=1:size(segs,1)
    st = segs(ii,1) ./ fs;
    fn = segs(ii,2) ./ fs;
    s = sprintf('[0:v]trim=start=%.3g:end=%.3g,setpts=PTS-STARTPTS[%gv];',...
        st,fn,ii-1);
    segstr = sprintf('%s%s',segstr,s);

    segstr2 = sprintf('%s[%gv]',segstr2,ii-1);
end

segstr2 = sprintf('%sconcat=n=%g:v=1[outv]',segstr2,ii);

% finish command
src = 'vid_18260994_full.mp4';
dst = 'trim_test2.mp4';
cmd = sprintf('ffmpeg -i %s -filter_complex "%s %s" -map [outv] %s',...
    src,segstr,segstr2,dst)
 
tic
unix(cmd)
toc

return
% make trimming string
segstr = 'select=''';
for ii=1:size(segs,1)
    %s = sprintf('[0:v]trim=start=%g:end=%g,
    st = segs(ii,1) ./ fs;
    fn = segs(ii,2) ./ fs;
    s = sprintf("between(t,%.3g,%.3g)+",st,fn);
    segstr = sprintf('%s%s',segstr,s);
end

segstr = sprintf("""%s',setpts=N/FRAME_RATE/TB""",segstr(1:end-1));

% finish command
src = 'vid_18260994_full.mp4';
dst = 'trim_test.mp4';
cmd = sprintf("ffmpeg -i %s -vf %s %s",src,segstr,dst);

tic
unix(cmd)
toc
