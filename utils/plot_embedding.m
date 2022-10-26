function out = plot_embedding(outClust,Lbnds,Ld,plotPeaks,assertAlphaData)
% out = plot_embedding(outClust,Lbnds,Ld)
% out = plot_embedding(outClust,Lbnds,Ld,plotPeaks)
% out = plot_embedding(outClust,Lbnds,Ld,plotPeaks,assertAlphaData)

% checks
if nargin < 4
    plotPeaks = [1 1];
else
    if numel(plotPeaks)==1 && plotPeaks==1
        plotPeaks = [1 1];
    end
end
if nargin < 5
    assertAlphaData = 1;
end

% info
labels = unique(Ld(:));
labels(labels==0) = [];
nlabel = max(labels);

d = outClust.dens2;
xv = outClust.xv;
yv = outClust.yv;

% plot density and cluster boundaries
[icol,irow] = find(Lbnds==1);
if assertAlphaData
    himg = imagesc(xv,yv,d,'alphadata',d>0);
else
    himg = imagesc(xv,yv,d);
end
hold all
hbnd = plot(xv(irow),yv(icol),'k.','MarkerSize',2);

% plot peaks
if any(plotPeaks)
    ht = nan(nlabel,1);
    hp = nan(nlabel,1);

    for ic=1:nlabel
        % data
        if 0
            mask = Ld==ic;
            tmp = d;
            tmp(~mask) = nan;

            [icol,irow] = find(tmp==max(tmp(:)));
        else
            [icol,irow] = find(Ld==ic);
        end

        icol = floor(median(icol));
        irow = floor(median(irow));
        xy = double([xv(irow) yv(icol)]);
        
        if plotPeaks(1); hp(ic) = plot(xy(1),xy(2),'r.','markersize',10); end
        if plotPeaks(2); ht(ic) = text(xy(1),xy(2),num2str(ic),'fontsize',14); end
    end
else
    hp = nan;
    ht = nan;
end

colorbar
axis square
xlabel('dim1')
ylabel('dim2')

% output
out = [];
out.hax = gca;
out.himg = himg;
out.hbnd = hbnd;
out.hpeak = hp;
out.htext = ht;

