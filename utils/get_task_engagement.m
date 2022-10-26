function [c,f,evt,info] = get_task_engagement(datfold,len,theseFrames)
% [c,f] = get_task_engagement(datfold,len)
% [c,f] = get_task_engagement({evt,info},len)
% [c,f] = get_task_engagement(...,theseFrames)
% [c,f,evt,info] = get_task_engagement(...)
    
% check
if nargin < 3
    theseFrames = [];
end

% load files
if ischar(datfold)
    load([datfold '/evt.mat'])
    load([datfold '/info.mat'])
else
    evt = datfold{1};
    info = datfold{2};
end

% data
nframe = floor(diff(info.firstLastFrame)/30000*30);
selevt = evt.leverPress==1;
    
if len == 0 % just sum lever presses
    c = sum(selevt) ./ nframe;
    f = nan;
else % engagemnt time series
    f_new= 0:nframe;
    f_evt = round(evt.frame(selevt));

    sel = ismember(f_new,f_evt);

    a = zeros(size(f_new));
    a(sel) = 1;

    % smooth it out
    win = ones(1,len);

    c = conv(a,win,'same');
    n = conv(ones(size(c)),win,'same');
    c = c ./ n;

    f = f_new;
    
    % cull if need be
    if ~isempty(theseFrames)
        [~,ia,ib] = intersect(f,theseFrames);
        f = f(ia);
        c = c(ia);
    end
end

%figure; plot(c)
