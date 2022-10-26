function T2 = unfurl_tree(T,ndat)
% T2 = unfurl_tree(T,ndat)

% T = z;
% ndat = size(a,1);

% nsplit
nsplit = max(max(T(:,1:2)));

% unfurl each split
thisMap = {};
for im=ndat+1:nsplit
    im2 = mod(im,ndat);
    tmp = T(im2,1:2);
    
    % check if its one of the already completed splits
    if all(tmp<=ndat)
        allLeaves = tmp;
    else
        allLeaves = [];
        for ii=1:2
            tmp = T(im2,ii);
            if tmp > ndat
                allLeaves = cat(2,allLeaves,thisMap{tmp-ndat});
            else
                allLeaves = cat(2,allLeaves,tmp);
            end
        end
    end
    thisMap{im-ndat} = allLeaves;
end

% replace elements of tree with all leaves
T2 = cell(size(T));
for it=1:size(T,1)
    for ii=1:2
        im = T(it,ii);
        if im <= ndat
            T2{it,ii} = im;
        else
            T2{it,ii} = thisMap{im-ndat};
        end
    end
    T2{it,3} = T(it,3);
end