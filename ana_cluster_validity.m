%load features and data

featdir = [anadir '/X_feat'];
if ~exist(featdir); mkdir(featdir); end
featdir_norm = [featdir '_norm'];
if ~exist(featdir_norm); mkdir(featdir_norm); end
    
infopath = [anadir '/info.mat'];

% choose training index
if strcmp(monk,'yo')
    idx_train2 = idx_train;
else
    idx_train2 = 1:5:numel(C);
end

% load features
if 1
    [X_feat,tmpdat] = load_features(featdir_norm,'feat',datasets);

    X_feat_train = X_feat(idx_train2,:);
    C_train = C(idx_train2);
end

% load distances


%% cluster validity
if 0
    fprintf('getting vluster validity...\n')

    nrand = 20;
    So = [];
    Sr = [];

    tic
    for id=1:numel(datasets)
        fprintf('%g,',id)
        seld = idat(idx_train2)==id;
        x = X_feat_train(seld,:);
        c = C_train(seld);

        % observed
        eva = evalclusters(x,c,'DaviesBouldin');
        So(id) = eva.CriterionValues;

        % random
        for ir=1:nrand
            cr = c(randperm(numel(c)));

            eva = evalclusters(x,cr,'DaviesBouldin');
            Sr(id,ir) = eva.CriterionValues;
        end
    end
    toc
end


% get means
[mu,se] = avganderror(So,'mean');
[mur,ser] = avganderror(mean(Sr,2),'mean');

m = [mu,mur];
s = [se ser];

figure
hb = barwitherr(s,m);
set(hb,'facecolor',[1 1 1]*0.5)
set(gca,'xticklabel',{'obs','rand'});

s = sprintf('cluster validity\nobs=%.3g+%.3g\nrand=%.3g+%.3g',mu,se,mur,ser);
title(s)
ylabel('mean DaviesBouldin')

sname = [figdir '/embedding_validity.pdf'];
save2pdf(sname,gcf)
    
%% cluster validity as a function of peak distance
if 1
    fprintf('clustervalidity, as a function of distance...\n')

    nrand = 20;
    cmb = combnk(1:max(C),2);
    ncmb = size(cmb,1);

    Spo = nan(numel(datasets),ncmb);
    Spr = nan(numel(datasets),ncmb,nrand);

    % loop over each combo of peaks
    tic
    parfor id=1:numel(datasets)
        try
        fprintf('%g,',id)
        for ic=1:ncmb
            sel = ismember(C_train,cmb(ic,:));
            sel = sel & idat(idx_train2)==id;
            x = X_feat_train(sel,:);
            c = C_train(sel);
            if numel(unique(c)) < 2
                continue
            end

            % observed
            eva = evalclusters(x,c,'DaviesBouldin');
            Spo(id,ic) = eva.CriterionValues;

            % random
            for ir=1:nrand
                cr = c(randperm(numel(c)));

                eva = evalclusters(x,cr,'DaviesBouldin');
                Spr(id,ic,ir) = eva.CriterionValues;
            end
        end  
        catch err
            warning('error: id=%g, ic=%g',id,ic)
            rethrow(err)
        end
    end
    fprintf('\n')
    toc
end


% get peaks
nlabel = max(C_train);
Ld = cluster_train.Ld;
xv = cluster_train.outClust.xv;
yv = cluster_train.outClust.yv;

peaks = [];
for ic=1:nlabel

    [icol,irow] = find(Ld==ic);

    icol = floor(median(icol));
    irow = floor(median(irow));
    xy = double([xv(irow) yv(icol)]);
    peaks(ic,:) = xy;
end

% distance between peaks, resort on the basis of distance

dpeaks = pdist(peaks);
dpeaks = squareform(dpeaks);

idx = sub2ind(cmb(:,1),cmb(:,2));
d = dpeaks(idx);
%d(end) = [];

[d,is] = sort(d);
Spo2 = Spo(:,is);
Spr2 = Spr(:,is,:);

% plot
ds = Spo2 - Spr2;

[mu,se] = avganderror(Spo2,'mean');
[mur,ser] = avganderror( nanmean(Spr2,3) ,'mean');
[mud,sed] = avganderror( nanmean(ds,3) ,'mean');

s = sum(Spr<Spo,3) ./ nrand;
p = s<0.05;
p = nanmean(p,2);
[mup,sep] = avganderror(p,'mean');

figure

scatter(d,mud)
lsline
pcl('y',0)

s = sprintf('cluster-pair validity as a function of embedding distances\nmean prop sig seperation=%.3g+%.3g',...
    mup,sep);
title(s)
ylabel('mean DaviesBouldin diff (=obs-rand)')
xlabel('embedding distance')
