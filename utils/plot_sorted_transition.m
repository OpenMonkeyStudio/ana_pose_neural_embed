function out = plot_sorted_transition(p,lab_orig,doSort)
% out = plot_sorted_transition(p,lab)
% out = plot_sorted_transition(p,lab,doSort)
%

% checks
if nargin < 3
    doSort = 1;
end

% prep
if doSort
    [lab,is] = sort(lab_orig);

    p2 = p(:,is);
    p2 = p2(is,:);
    str = 'Sorted State';
else
    is = [];
    p2 = p;
    lab = lab_orig;
    str = 'State';
end

xx = 1:size(p2,1);

% plot
h = imagesc(xx,xx,p2);

xlabel([str ' (t)'])
ylabel([str ' (t+1)'])
colorbar
axis square
hold all

% bounding boxes
if doSort
    st = 1;
    hb = [];
    for il=1:max(lab)
        s = lab(st);
        fn = find(lab==s,1,'last');
        st = st-0.5;
        fn = fn+0.5;

        xx = [st fn fn st st];
        yy = [st st fn fn st];
        hb(il) = plot(xx,yy,'k-','linewidth',2);
        st = fn+0.5;
    end
    hold off
else
    hb = [];
end

% output
out = [];
out.p_orig = p;
out.label_orig = lab_orig;
out.p_sorted = p2;
out.label_sorted = lab;
out.isort = is;
out.hax = gca;
out.h_boxes = hb;
out.h_mat = h;