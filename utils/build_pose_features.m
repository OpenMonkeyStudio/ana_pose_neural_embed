function [X_out,feat_labels,ifeat,procInfo] = build_pose_features_new(data,theseFeatures,procInfo_in)
%  [X_out,feat_labels,ifeat,procInfo] = build_pose_features_new(data,theseFeatures,procInfo_in)

%% prep
data = checkfield(data,'fs',30);
data = checkfield(data,{'X','data'},'needit');
data = checkfield(data,'frame','needit');
data = checkfield(data,'idat',ones(numel(data.frame),1));
data = checkfield(data,'normalization',{});
data = checkfield(data,'id_pca',1);
data = checkfield(data,'cfg_wvt',[]);
data = checkfield(data,'cfg_pca',[]);
data = checkfield(data,'cfg_smooth',[]);

if nargin < 3 || isempty(procInfo_in)
    useOldProc = 0;
    procInfo_in = {};
else
    useOldProc = 1;
end

% prep dataset IDs. select which dataet to process first here
[udat,~,idat] = unique(data.idat);

sel = data.idat==data.id_pca;
id1 = unique(idat(sel));
newval = [id1 1];
oldval = [1 id1];
idat = changem(idat,newval,oldval);

% backwards comptability
if ~isfield(data,'spine')
    data.spine = data.neck - data.hip;
end


% wavelet params
%foi = [0.1:0.1:data.fs/2];
%foi = [0.01:0.01:0.1, 0.2:0.1:5, 5.5:0.5:data.fs/2];
%foi = [0.2:0.1:5, 5.5:0.5:data.fs/2];
%foi = [0.1:0.1:5, 5.5:0.5:data.fs/2];
foi = logspace(log(0.1)./log(10), log(data.fs/2)./log(10),100);
%foi = linspace(0.1,data.fs/2,100);


wcfg = data.cfg_wvt;
wcfg = checkfield(wcfg,'method','tfr'); % mtmconvol
wcfg = checkfield(wcfg,'fs',30);
wcfg = checkfield(wcfg,'width',5);
wcfg = checkfield(wcfg,'gwidth',3);
wcfg = checkfield(wcfg,'foi',foi);
wcfg = checkfield(wcfg,'pc_dims',20);
wcfg = checkfield(wcfg,'doImpute',1);
wcfg = checkfield(wcfg,'diffWhiten',1);

% straight PCA params
pcfg = data.cfg_pca;
pcfg = checkfield(pcfg,'pc_dims',15);
pcfg = checkfield(pcfg,'pc_thresh',95);

% smoothing params
scfg = data.cfg_smooth;
scfg = checkfield(scfg,'smoothLens',[10 30 60]);
scfg = checkfield(scfg,'win','gauss');

if strcmp(scfg.win,'gauss')
    scfg.smoothWins = cellfun(@(x) gausswin(x),num2cell(scfg.smoothLens),'un',0);
else
    error('which win?')
end

%% call each function
procInfo = {};

X_out = [];
ifeat = {};
feat_labels = {};
for ii=1:numel(theseFeatures)
    feat = theseFeatures{ii};
    fprintf('feature: %s\n',feat)
    
    procInfo{ii,1} = feat;
    
    % parse
    tok = split(feat,'_');
    inputType = tok{1};    
    if numel(tok)>1
        procs = tok(2:end);
    else
        procs = {};
    end
    
    if ~useOldProc
        procSteps = cell(1,numel(procs));
    else
        selproc = strcmp(procInfo_in(:,1),feat);
        if sum(selproc)==0
            error('missing procs in old info')
        end
      
        procSteps = procInfo_in{selproc,2};
    end
    
    % loop over each dataset and apply the procs
    xout = nan(size(data.X,1),1);
    for id=1:numel(udat)
        seldat = idat==id;
        
        % prep inputs
        data2 = data;
        data2.X(~seldat,:) = [];
        data2.frame(~seldat,:) = [];
        data2.idat(~seldat,:) = [];
        data2.com(~seldat,:) = [];
        data2.spine(~seldat,:) = [];
        [data2,prefix] = parse_inputs(data2,inputType);
        x = data2.X;
        
        % apply processing
        suffix = '';
        for ip=1:numel(procs)
            p = split(procs{ip},'-');
            
            if numel(p)>1; p_input = p(2:end); else; p_input={}; end
            p = p{1};
            
            switch p
                case 'pca'
                    newsuffix = 'Pca';
                    if ~useOldProc                        
                        if id==1
                            pcfg2 = pcfg;
                            [pcfg2.c,x,~,~,pcfg2.v,pcfg2.mu] = pca(x);
                        else
                            x = (x-pcfg2.mu) * pcfg2.c;
                        end
                    else
                        pcfg2 = procSteps{ip};
                        x = (x-pcfg2.mu) * pcfg2.c;
                    end
                    
                    n = min(pcfg2.pc_dims,size(x,2));
                    x = x(:,1:n);
                    data2.labels_new = cellfun(@(x) sprintf('pc%g',x),num2cell(1:n),'un',0);
                    procSteps{ip} = pcfg2;
                case 'wvt'
                    newsuffix = 'Wvt';
                    if ~useOldProc                        
                        wcfg2 = wcfg;
                        wcfg2.doPCA = 0;
                        wcfg2.labels = data2.labels_new;
                    else
                        wcfg2 = procSteps{ip};
                    end
                    [x,wcfg2] = wvt_pca(wcfg2,x,data2.time);
                    procSteps{ip} = wcfg2;
                case 'wvtpca'
                    newsuffix = 'Wvtpca';
                    if ~useOldProc                        
                        if id==1 % compute pca
                            wcfg2 = wcfg;
                            wcfg2.doPCA = 1;
                        else % apply from first dataset
                            wcfg2.doPCA = 2;
                        end
                        wcfg2.labels = data2.labels_new;
                    else
                        wcfg2 = procSteps{ip};
                        wcfg2.doPCA = 2;
                    end
                    [x,wcfg2] = wvt_pca(wcfg2,x,data2.time);
                    procSteps{ip} = wcfg2;
                case 'filter'
                    tmpdat = freqfilter(x',data2.time,data2.fs,[0.05 Inf]);
                    x = tmpdat.trial{1}';
                    newsuffix = 'flt';
                case 'smooth'
                    x = apply_smoothing(x,scfg.smoothWins);
                    newsuffix = 'sm';
                case 'movstd'
                    x = get_movestd(x,scfg.smoothLens);
                    newsuffix = 'mvSd';
                case 'movmean'
                    x = get_movemean(x,scfg.smoothLens);
                    newsuffix = 'mvMu';
                case 'movrange'
                    x = get_moverng(x,scfg.smoothLens);
                    newsuffix = 'mvRng';
                case 'speed'
                    x = speed(x,data2.time,data2.labels_new);
                    newsuffix = 'sp';
                case 'velocity'
                    x = velocity(x,data2.time);
                    newsuffix = 'vel';
                case 'interstd'
                    x = nanstd(x,[],2);c
                    newsuffix = 'inSd';
                case 'intermean'
                    x = nanmean(x,2);
                    newsuffix = 'inMu';
                case 'rangexyz'
                    x = get_rangexyz(x,data2.labels_new);
                    newsuffix = 'xyzRng';
                    foo=1;
                case 'rot2spine'
                    x = rotate2spine(x,data2.labels_new,data2.spine);
                    foo=1;
                case 'gravityangle'
                    normal = [0 1 0];
                    v1 = x(:,contains(data2.labels_new,'neck'));
                    v2 = x(:,contains(data2.labels_new,'hip'));
                    v = v1 - v2;
                    u = repmat(normal,size(v,1),1);
                    
                    a = acos(dot(u,v,2) ./ (vecnorm(u,2,2) .* vecnorm(v,2,2)));
                    
                    foo=1;
                case 'perpendicularity'
                    %{
                    if strcmp(p_input{1},'spine')
                        neck = data2.X(:,contains(data2.labels_new,'neck'));
                        hip = data2.X(:,contains(data2.labels_new,'hip'));
                        ref0 = neck;
                        ref = hip - neck;
                    else
                        error('huh')
                    end
                                        
                    lab = cellfun(@(x) x(1:end-2),data2.labels_new,'un',0);
                    lab(contains(lab,{'neck','hip'})) = [];
                    [ulab,~,ilab] = unique(lab);
                    
                    x = [];
                    for ig=1:numel(ulab)
                        a = data2.X(:,contains(data2.labels_new,ulab{ig}));
                        a = a-ref0;
                        tmp = perpendicularity(a,ref);
                        x= cat(2,x,tmp);
                    end
                    %}
                    
                    jg = data2.joint_graph;
                    cmb = combnk(1:size(jg,1),2);
                    x2 = [];
                    newlab = {};
                    for ii=1:size(cmb,1)
                        V = [];
                        for jj=1:2
                            v1 = x(:,contains(data2.labels_new,jg{cmb(ii,jj),1}));
                            v2 = x(:,contains(data2.labels_new,jg{cmb(ii,jj),2}));
                            V(:,:,jj) = v2-v1;
                            if jj==1; ref = v1;
                            else; V(:,:,jj) =  V(:,:,jj)-ref;
                            end
                        end

                        tmp = perpendicularity(V(:,:,1),V(:,:,2));
                        x2 = cat(2,x2,tmp);
                        s = sprintf('%s-%s:%s-%s',jg{cmb(ii,1),1},jg{cmb(ii,1),2},jg{cmb(ii,2),1},jg{cmb(ii,2),2});
                        newlab = cat(2,newlab,s);
                    end
                    x = x2;
                    newsuffix = 'segPerp';
                case 'segmentlength'
                    [x,newlab] = segment_length(data2.X,data2.labels_new,data2.joint_graph);
                    data2.labels_new = newlab;
                    newsuffix = 'segL';
                    foo=1;
                case 'relativelength'
                    [x,newlab] = relative_length(data2.X,data2.labels_new,data2.joint_graph);
                    data2.labels_new = newlab;
                    newsuffix = 'relL';
                    foo=1;
                case 'relativeangle'
                    [x,newlab] = relative_angle(data2.X,data2.labels_new,data2.joint_graph);
                    data2.labels_new = newlab;
                    newsuffix = 'relAng';
                    foo=1;
                case 'cart'
                    xx = split(p_input{1},'');
                    xx([1 end]) = [];
                    
                    x = [];
                    labs = {};
                    for ix=1:numel(xx)
                        sel = contains(data2.labels_new,['_' xx{ix}]);
                        x = cat(2,x,data2.X(:,sel));
                        labs = cat(2,labs,data2.labels_new(sel));
                    end
                    data2.labels_new = labs;
                    newsuffix = p_input{1};
                    foo=1;
                otherwise
                    error('dont recognize: %s, %s',inputType,p)
            end
            
            suffix = [suffix '-' newsuffix];
        end
        
        % append
        procInfo{ii,2} = procSteps;
        
        if id==1
           xout = repmat(xout,1,size(x,2)); % now we know what  the final dimensionality is
           feats = cellfun(@(x) [prefix suffix '_' num2str(x)], num2cell(1:size(xout,2)),'un',0);
        end
        xout(seldat,:) = x;
    end
    
    % findla apend
    ifeat = [ifeat,[1:size(xout,2)] + size(X_out,2)];
    X_out = cat(2,X_out,xout);
    feat_labels = cat(2,feat_labels,feats);
end


%% normalize
clipThresh = Inf;
    
if ~isempty(data.normalization)
    % indep normalization
    for ii=1:size(data.normalization,1)
        normType = data.normalization{ii,1};
        fprintf('%s normalization \n',normType)
        
        switch normType
            case {'indep_zscore','indep_rng'}
                s = data.normalization{ii,2};
                if ~isempty(s); idx = cat(1,ifeat{ismember(theseFeatures,s)});
                else; idx = 1:size(X_out,2);
                end
                
                tmp = X_out(:,idx);
                switch normType
                    case 'indep_zscore'
                        if 1
                            if 0
                                mu = nanmedian(tmp);
                                se = mad(tmp,1,1)*1.4826;
                            else
                                mu = nanmedian(tmp(idat==1,:));
                                se = mad(tmp(idat==1,:),1,1)*1.4826;
                            end
                        else
                            mu = nanmean(tmp);
                            se = nanstd(tmp);
                            %se(se==0) = eps;
                        end
                        X_out(:,idx) = (X_out(:,idx)-mu) ./ se;
                        
                        % clip
                        if 1
                            X_out(:,idx) = max(min(X_out(:,idx),clipThresh),-clipThresh);
                        end
                    case 'indep_rng'
                        tmp = X_out(:,~ignore);
                        mn = min(tmp);
                        mx = max(tmp);
                        X_out(:,~ignore) = (tmp-mn)./(mx-mn);
                end
            case 'set_zscore'
                s = data.normalization{ii,2};
                selfeat = find( ismember(theseFeatures,s) );
                
                for jj1=1:numel(selfeat)
                    % norm
                    jj = selfeat(jj1);
                    tmp = X_out(:,ifeat{jj});
                    mu = nanmedian([tmp(:)]);
                    se = mad([tmp(:)],1,1)*1.4826;
                    tmp = (tmp-mu)./se;
                    
                    % clip
                    if 1
                        mn = min([tmp(:)],clipThresh);
                        mx = max(mn,-clipThresh);
                        tmp2 = reshape(mx,size(tmp));
                    end
                
                    % update
                    X_out(:,ifeat{jj}) = tmp;
                end
                

            case 'reweightSet'
                % reweight each set now
                nset = numel(ifeat);
                for jj=1:nset
                    idx = ifeat{jj};
                    tmp = X_out(:,idx);
                    w = 1./(nset*size(tmp,2));
                    tmp = tmp * w;
                    X_out(:,idx) = tmp;
                end
        end
    end
end


%% final

% get rid of bad features
if 1
    sel = isnan(X_out) | abs(X_out)==Inf | all(X_out==X_out(1,:),1);
    bad = all(sel);
    X_out(:,bad) = [];
    feat_labels(bad) = [];

    ibad = find(bad);
    ifeattmp = ifeat;
    for ii=1:numel(ifeat)
        if ii==7;
            foo=1;
        end
        del = ismember(ifeat{ii},ibad);
        ifeattmp{ii}(del) = [];

        % change the other indices
        if sum(del)>0
            ifeattmp{ii} = 1:numel(ifeattmp{ii});
            if ii>1; ifeattmp{ii} = ifeattmp{ii}+ifeattmp{ii-1}(end); end
            if ii<numel(ifeattmp)
                for jj=ii+1:numel(ifeattmp)
                    ifeattmp{jj} = ifeattmp{jj}-ifeattmp{jj}(1)+ifeattmp{jj-1}(end)+1;
                end
            end
        end
    end
end

foo=1;
%% FUNCS

% /////////////////////////////////////////////////////////////////
%                       PROCESSING funcs
% /////////////////////////////////////////////////////////////////

function x2 = rotate2spine(x,labels,spine_orig)
    neck = x(:,contains(labels,'neck'));
    hip = x(:,contains(labels,'hip'));
    
    spine = neck - hip;
    
    x2 = x - repmat(hip,1,numel(labels)/3);

    % Find rotation matrix
    Z_dir = spine_orig;
    Y_dir = cross(Z_dir,-spine,2);
    X_dir = cross(Y_dir,Z_dir,2);
    
    e1 = X_dir ./ vecnorm(X_dir,2,2);
    e2 = Y_dir ./ vecnorm(Y_dir,2,2);
    e3 = Z_dir ./ vecnorm(Z_dir,2,2);
    R = cat(3,e1,e2,e3);
    R = permute(R,[2 3 1]);

    for is=1:size(x,1)
        r = R(:,:,is);
        x2(is,:) = rotate_dataset(x2(is,:),r);        
    end 
   
    foo=1;
    
    %{
    
    %s1 = spine_orig(130,:);
    %s2 = spine(130,:);
    s1 = spine_orig(130,:);
    %s2 = x2(130,contains(labels,'neck'))-x2(130,contains(labels,'hip'));
    s2 = x2(130,contains(labels,'neck'))-x2(130,contains(labels,'hip'));
    s3 = spine(130,:);
    s1 = [[0 0 0]; s1]; 
    s2 = [[0 0 0]; s2]; 
    s3 = [[0 0 0]; s3]; 
    figure; plot3(s1(:,1),s1(:,2),s1(:,3),'r'); 
    hold all; 
    plot3(s2(:,1),s2(:,2),s2(:,3),'b');
    plot3(s3(:,1),s3(:,2),s3(:,3),'g');
    
    %}
    
foo=1;
function x2 = get_rangexyz(x,labels)
    str = {'_x','_y','_z'};
    
    x2 = [];
    for ii=1:3
        sel = contains(labels,str{ii});
        tmp = x(:,sel);
        mx = max(tmp,[],2);
        mn = min(tmp,[],2);
        x2 = cat(2,x2,mx-mn);
    end
    foo=1;
    
function x2 = apply_smoothing(x,smoothWins)

    x2 = nan(size(x,1), size(x,2)*numel(smoothWins)); 

    for ii=1:numel(smoothWins)
        win = smoothWins{ii};

        c = conv2(win,1,x,'same');
        n = conv(ones(size(x,1),1),win,'same');
        tmp = c ./ n;
        %tmp = (tmp-min(tmp(:))) ./ (max(tmp(:))-min(tmp(:)));
        idx = [1:size(x,2)] + (ii-1)*size(x,2);
        x2(:,idx) = tmp;
    end


function x2 = apply_medfilt(x,smoothWins)
    x2 = nan(size(x,1), size(x,2)*numel(smoothWins)); 

    for ii=1:numel(smoothWins)
        L = numel(smoothWins(ii));
    
        tmp = medfilt1(x,L,[],1,'truncate');
        idx = [1:size(x,2)] + (ii-1)*size(x,2);
        x2(:,idx) = tmp;
    end

function x2 = get_movemean(x,smoothLengths)
    x2 = nan(size(x,1), size(x,2)*numel(smoothLengths)); 
    for ii=1:numel(smoothLengths)
        L = smoothLengths(ii);

        tmp = movmean(x,L,1);
        idx = [1:size(x,2)] + (ii-1)*size(x,2);
        x2(:,idx) = tmp;
    end          

function x2 = get_moverng(x,smoothLengths)
    x2 = nan(size(x,1), numel(smoothLengths)); 
    for ii=1:numel(smoothLengths)
        L = smoothLengths(ii);

        mx = movmax(max(x,[],2),L,1);
        mn = movmin(min(x,[],2),L,1);
        r = mx-mn;
        x2(:,ii) = r;
    end 
    
function x2 = get_movestd(x,smoothLengths)
    x2 = nan(size(x,1), size(x,2)*numel(smoothLengths)); 
    for ii=1:numel(smoothLengths)
        L = smoothLengths(ii);

        tmp = movstd(x,L,1);
        idx = [1:size(x,2)] + (ii-1)*size(x,2);
        x2(:,idx) = tmp;
    end          
 

function tmpdat = freqfilter(x,time,fs,pf,lab)
    if nargin < 5
        lab = cellfun(@(x) num2str(x),num2cell(1:size(x,1)),'un',0);
    end

    % fill in nans
    for ix=1:size(x,1)
        tmpx = x(ix,:);
        bad = isnan(tmpx);
        vq = interp1(time(~bad),tmpx(~bad),time(bad),'pchip','extrap');
        x(ix,bad) = vq;
    end
    
    % highpass filter to help with non-stationarity
    tmpdat = [];
    tmpdat.trial = {x};
    tmpdat.time = {time};
    tmpdat.label = lab;
    tmpdat.fsample = fs;

    pcfg = [];
    ft = 'firws';

    if all(abs(pf)==Inf)
        
    elseif ~any(abs(pf)==Inf) % bandpass
        pcfg.bpfilter = 'yes';
        pcfg.bpfreq = pf;
        pcfg.bpfilttype = ft;
    else
        % high pass
        if abs(pf(2))==Inf && abs(pf(1))~=Inf
            pcfg.hpfilter = 'yes';
            pcfg.hpfreq = pf(1);
            pcfg.hpfilttype = ft;
        end

        % low pass
        if abs(pf(1))==Inf && ~abs(pf(2))~=Inf
            pcfg.lpfilter = 'yes';
            pcfg.lpfreq = pf(2);
            pcfg.lpfilttype = ft;
        end
    end
    [tmpdat] = ft_preprocessing(pcfg, tmpdat);
    foo=1;

function [y,cfg] = wvt_pca(cfg,x,time)

    % prep
    doImpute = cfg.doImpute;
    diffWhiten = cfg.diffWhiten;
    
    x = x';
    lab = cellfun(@num2str,num2cell(1:size(x,1)),'un',0);
    %lab = cfg.labels;

    xorig = x;
    
    % impute values?
    if doImpute
        % orig
        time1 = time;
        x1 = x;
        
        % new time axis
        time = 0:time1(end)*cfg.fs;
        tmp = ismember(time, round(time1*cfg.fs));
        goodtime = find(tmp);
        badtime = find(~tmp);
        x2 = nan(size(x,1),numel(time));
        for ix=1:size(x,1)
            tmp = nan(size(time));
            tmp(goodtime) = x1(ix,:);
            tmp(badtime) = interp1(time1,x1(ix,:),time(badtime)./cfg.fs,'linear');

            mu = tmp(goodtime(1));
            tmp(1:goodtime(1)-1) = mu;            
            x2(ix,:) = tmp;
        end
        
        % udpate
        time = time ./ cfg.fs;
        x = x2;
    end
    
    % whiten by differentiation
    if diffWhiten
        if 1
            [b,g] = sgolay(3,5);
            dt = 1./cfg.fs;
            p = 1;
            f = factorial(p)/(-dt)^p * g(:,p+1);
            f = f./sum(abs(f));
            x = conv2(f,1,x','same')';
        else
            if 0
                x = sgolayfilt(x,3,5,[],2);
            end
            x = [zeros(size(x,1),1), diff(x,[],2)]; % whiten
        end
    end
    
    
    % running normalization
    if 0
        nwin = 1*30+1;
        if 0
            xs = movsum(abs(x),nwin,2);
            x = x ./ xs;
        elseif 0
            mu = movmean(x,nwin,2);
            se = movstd(x,nwin,[],2);
            x2 = (x).*se;
        end
    end
    
    % filter
    time2 = [0:numel(time)-1]./cfg.fs;
    tmpdat = freqfilter(x,time2,cfg.fs,[Inf Inf],lab);
    %tmpdat = freqfilter(x,time2,cfg.fs,[cfg.foi(1)/2 Inf],lab);

    % freq analysis
    tmp = split(cfg.method,'_');
    method = tmp{1};
    if numel(tmp)>1; method2 = tmp{2}; end
    
    switch method
        case 'pyramid'
            ipyr = [1 2:2:2^4];
            for ip=1:numel(ipyr)
                step = ipyr(ip);
                idx = 1:step:numel(time);
                xtmp = x(:,idx);
                ttmp = time(idx); %[0:size(xtmp,2)] ./ cfg.fs;
                
                cfg2 = cfg;
                cfg2.doImpute = 0;
                cfg2.diffWhiten = 0;
                cfg2.method = method2;
                cfg2.doPCA = 0;
                cfg2.fs = cfg.fs ./ step;
                
                [ytmp,cfg2] = wvt_pca(cfg2,xtmp',ttmp);
            end
            
        case {'mtmconvol','superlet','wavelet','tfr'}
            pcfg              = [];
            pcfg.output       = 'fourier';
            pcfg.taper        = 'hanning';
            pcfg.foi          = cfg.foi;
            pcfg.toi = time2;
            pcfg.pad = 'nextpow2';
            switch method
                case 'mtmconvol'
                    pcfg.method = 'mtmconvol';
                    pcfg.t_ftimwin    = cfg.width./pcfg.foi;
                    pcfg.tapsmofrq = pcfg.foi/4;
                    pcfg.correctt_ftimwin = 'yes';
                case 'superlet'
                    pcfg.method = 'superlet';
                    pcfg.superlet.basewidth = cfg.width;
                    pcfg.superlet.gwidth = cfg.gwidth;
                case 'wavelet'
                    pcfg.method = 'wavelet';
                    pcfg.width = cfg.width;
                    pcfg.gwidth = cfg.gwidth;
                case 'tfr'
                    pcfg.method = 'tfr';
                    pcfg.width = cfg.width;
                    pcfg.gwidth = cfg.gwidth;
                otherwise
                    errror('which spectral method?')
            end
            ftmp = ft_freqanalysis(pcfg, tmpdat);
        case 'hilbert'
            tic
            ftmp = [];
            ftmp.freq = cfg.foi;
            ftmp.time = tmpdat.time{1};
            ftmp.fourierspctrm = nan(1,size(x,1),numel(cfg.foi),numel(tmpdat.time{1}));

            fprintf('bandpass/hilbert')
            for ii=1:numel(cfg.foi)
                dotdotdot(ii,0.1,numel(cfg.foi))
                Fbp = cfg.foi(ii) + [-cfg.foi(ii) cfg.foi(ii)]/4;
                Fbp = min(Fbp,cfg.fs/2);
                [filt] = ft_preproc_bandpassfilter(tmpdat.trial{1}, cfg.fs, Fbp, 4, 'but', 'twopass', 'reduce');

                s = hilbert(filt')';
                s = permute(s,[3 1 4 2]);
                ftmp.fourierspctrm(1,:,ii,:) = s;
            end
        case 'cwt'
            fb = cwtfilterbank('signallength',size(x,2),'wavelet','amor','samplingfrequency',cfg.fs);
            cfs = cwt(x(1,:));
            res =  zeros(size(x,1),size(cfs,1),size(cfs,2));

            tic
            for k=1:size(x,1)
                [res(k,:,:),f] = wt(fb,x(k,:));
            end
            toc

            % append
            ftmp = [];
            ftmp.freq = f;
            ftmp.time = tmpdat.time{1};
            ftmp.fourierspctrm = permute(res,[4 1 2 3]);
        case 'wavedec'
            nlevel = 4;
            
            res = nan(nlevel,numel(time));
            res2 = nan(numel(time),numel(cfg.foi)*size(x,1),nlevel);
            
            tmpx = nan(size(x,1)*nlevel,numel(time));
            itmp = 0;
            for ix=1:size(x,1)
                [c,l] = wavedec(x(ix,:),nlevel,'db2');
                dc = detcoef(c,l,'cells');
            
                for ii=1:numel(dc)
                    tmpc = dc{ii};
                    tmpt = linspace(0,time(end),numel(tmpc));

                    tmp = interp1(tmpt,tmpc,time,'linear');
                    
                    % append
                    itmp=itmp+1;
                    tmpx(itmp,:) = tmp;
                    %res(ii,:) = tmp;
                end
            end
            
            % now do fourier
            cfg2 = cfg;
            cfg2.doImpute = 0;
            cfg2.diffWhiten = 0;
            cfg2.method = 'tfr';
            cfg2.doPCA = 0;

            [ytmp,cfg2] = wvt_pca(cfg2,tmpx',time);
            ytmp2 = reshape(ytmp,[numel(time) numel(cfg.foi) nlevel*size(x,1)]);
            ytmp2 = permute(ytmp2,[4 3 2 1]);
            ytmp2 = sqrt(ytmp2);
            
            %{
            figure;
            nr = 3; nc = 4;
            for ii=1:numel(dc)
                lim = a/30+[-1 1]*20;
                selx = time >= lim(1) & time <= lim(2);
                
                subplot(nr,nc,ii); 
                imagesc2(time(selx),cfg.foi,res2(selx,:,ii)'); 
                plotcueline('x',a/30); 
                colorbar; 
            end
            %}
            foo=1;
            
            % append
            ftmp = [];
            ftmp.freq = cfg.foi;
            ftmp.time = tmpdat.time{1};
            ftmp.fourierspctrm = ytmp2;
    end
    
    if doImpute
        ftmp.fourierspctrm = ftmp.fourierspctrm(:,:,:,goodtime);
        ftmp.time = ftmp.time(goodtime);
        
        time = time(goodtime);
        x = x(:,goodtime);
    end
    cfg.foi = ftmp.freq;
    s = permute(ftmp.fourierspctrm,[2 3 4 1]);

    %{
    figure; 
    [nr,nc]=subplot_ratio(size(x,1));
    for ii=1:size(x,1)
        subplot(nr,nc,ii)
        tmp = x(ii,:) - mean(x(ii,:));
        plot(tmp)
        hold all
        plot(tmpdat.trial{1}(ii,:))
        title(lab{ii})
    end
    %}

    % power
    y = abs(s).^2;

    % whiten
    if 1 && ~diffWhiten
        if 0
            p = nanmedian(nanmedian(y,1),3);
            ps = sgolayfilt(p,3,5);
        else
            ps = 1./pcfg.foi;
        end
       
        %save('test1.mat','x','y','p','ps','tmpdat','s','ftmp')
        y = y ./ ps;
    end
    
    
    % reshape
    y = permute(y,[2 1 3]); % keep freqs together
    y = reshape(y,[size(y,1)*size(y,2), size(y,3)])';
    
    % get rid of nans, embedding doesnt like it
    for ii=1:size(y,2)
        st = find(~isnan(y(:,ii)),1);
        fn = find(~isnan(y(:,ii)),1,'last');

        if st > 1
            y(1:st-1,ii) = y(st,ii);
        end
        if fn < size(y,1)
            y(fn:end,ii) = y(fn,ii);
        end
    end

    if any(all(isnan(y)))
        error('some columsn are all nans...')
    end

    if 0
        y = zscore(y);
    end

    % clip outliers    
    if 1
        if ~isfield(cfg,'clip_thresh')
            tmp = [y(:)];
            mx = prctile(tmp,99.9);
            cfg.clip_thresh = mx;
        else
            mx = cfg.clip_thresh;
        end
        y = min(y,mx);
    end

    % now do PCA
    if cfg.doPCA>0
        if cfg.doPCA==1
            fprintf('computing pca, using first dataset...\n')
            y1 = y;
            
            if 1
                [wvt_c, y, ~, ~, vw, wvt_mu] = pca(y);
                %[~, pc_dims] = min(cumsum(vw) > cfg.thresh == 0);
                cfg.wvt_c = wvt_c;
                cfg.wvt_v = vw;
                cfg.wvt_mu = wvt_mu;
            else
                [U,S,V] = svds(y1,cfg.pc_dims);
                y = U*S;
            end
            
            %{
            figure; 
            [nr,nc] = subplot_ratio(20);
            xt = [1 numel(cfg.foi):numel(cfg.foi):size(y,2)];
            str = {'lhand-rhand'  'rhand-rfoot'  'rfoot-lfoot'  'lfoot-lhand'};
            for ii=1:20
                subplot(nr,nc,ii)
                c=wvt_c(:,ii);
                plot(c); 
                %plotcueline('x',a); 
                set(gca,'xtick',xt,'xlim',[1 size(y,2)])
                for jj=1:numel(str); ix=mean(xt(jj)); text(ix*1.1,min(get(gca,'ylim')),str{jj}); end
                grid on
            end
            setaxesparameter('ylim')
            %}
            foo=1;
        elseif cfg.doPCA==2
            fprintf('applying pca, using first dataset...\n')
            y=(y-cfg.wvt_mu) * cfg.wvt_c;
        end
        mn = min(cfg.pc_dims,size(y,2));
        y = y(:,1:mn);
    end

% /////////////////////////////////////////////////////////////////
%                       INPUT funcs
% /////////////////////////////////////////////////////////////////

function [data,prefix] = parse_inputs(data,inputType)

    % prep
    xstr = {'x','y','z'};
    
    X = data.X;
    com = data.com;
    ref_spine = data.spine;
    labels = data.labels;
    time = data.frame' ./ data.fs;
    %time = [1:numel(data.frame)]' ./ data.fs;
    njoints  = size(X,2)/3;

    data.time = time;
    
    % determine inputs
    tmp = split(inputType,'-');
    inputType = tmp{1};
    if numel(tmp)>1;types = tmp(2:end);
    else types = {};
    end
    
    if strcmp(inputType,'perpendicularity')
        a = repmat([0 1 0],size(ref_spine,1),1);
        data.X = perpendicularity(a,ref_spine);
        prefix = 'perp';
        newlab = {prefix};
        foo=1;
    elseif strcmp(inputType,'height')
        data.X = com(:,2);
        prefix = 'height';
        newlab = {'height'};
    elseif strcmp(inputType,'heightCategory')
        tmp = com(:,2);
        data.X = double(tmp>2)+1;
        prefix = 'heightCat';
        newlab = {'heightCat'};
    elseif strcmp(inputType,'comSpeed')
        data.X = speed(com,time,{'com_x','com_y','com_z'});
        prefix = 'comS';
        newlab = {'comSpeed'};
    elseif strcmp(inputType,'groundSpeed')
        data.X = speed(com(:,[1 3]),time,{'com_x','com_z'});
        prefix = 'groundS';
        newlab = {'groundSpeed'};
    elseif strcmp(inputType,'heightSpeed')
        data.X = speed(com(:,2),time,{'com_y'});
        prefix = 'heightS';
        newlab = {'heightSpeed'};
    elseif strcmp(inputType,'planarSpeed') % world coordinates
        tmpx = [];
        cmb = combnk(1:3,2);
        strs = {'com_x','com_y','com_z'};
        for ii=1:3
            idx = cmb(ii,:);
            tmp = speed(com(:,idx),time,strs(idx));
            tmpx = cat(2,tmpx,tmp);
        end
        data.X = tmpx;
        prefix = 'planeS';
        newlab = cellfun(@(x) ['planeS_' x],xstr,'un',0);
    elseif strcmp(inputType,'distanceSpine2hands')
        
        foo=1;
    elseif strcmp(inputType,'heightVelocity')
        data.X = velocity(com(:,2),time);
        prefix = 'heightV';
        newlab = {'heightV'};
    elseif strcmp(inputType,'planarVelocity') % world coordinates
        tmpx = [];
        cmb = combnk(1:3,2);
        strs = {};
        for ii=1:3
            idx = cmb(ii,:);
            tmp = velocity(com(:,idx),time);
            tmpx = cat(2,tmpx,tmp);
            strs{ii} = sprintf('%s%s',xstr{idx(1)},xstr{idx(2)});
        end
        data.X = tmpx;
        prefix = 'planeV';
        newlab = cellfun(@(x) num2str(x),num2cell(1:size(tmpx,2)),'un',0);
    elseif contains(inputType,'hands') % get limb metrics
        ii = strfind(inputType,'-'); 
        if isempty(ii); ii = numel(inputType); else; ii = ii(1); end
        m = lower(inputType(6:ii));
        [data.X,newlab] = hand_metrics(X,labels,m);
        prefix = 'handsDistance';
    
    elseif strcmp(inputType,'landmark')
        sel = contains(data.labels,types);
        data.X = X(:,sel);
        newlab = data.labels(sel);
        prefix = [types{:}];
    elseif contains(inputType,'limb') % get limb metrics
        % get the appropripriate graph
        switch inputType(1:5)
            case 'limb1'; [data.limbs,data.joint_graph] = limb_graph1(njoints);
            case 'limb2'; [data.limbs,data.joint_graph] = limb_graph2();
            case 'limb3'; [data.limbs,data.joint_graph] = limb_graph3();
            case 'limb4'; [data.limbs,data.joint_graph] = limb_graph4();
            case 'limb5'; [data.limbs,data.joint_graph,X] = limb_graph5(X,labels);
        end
        
        % cull the data
        ulab = unique([data.limbs(:)]);
        bad = ~contains(labels,ulab);
        X(:,bad) = [];
        labels(bad) = [];
        
        % get the metric
        ii = strfind(inputType,'-'); 
        if isempty(ii); ii = numel(inputType); else; ii = ii(1); end
        metric = lower(inputType(6:ii));
        if numel(inputType)>5
            [data.X,data.pairsLabels] = limb_metrics(X,labels,data.limbs,metric);
            prefix = inputType(1:min(numel(inputType),9));
        else
            data.X = X;
            data.pairsLabels = labels;
            prefix = inputType(1:5);
        end
       
        newlab = data.pairsLabels;
    elseif strcmp(inputType,'spine')
        tmp = {'neck','hip'};
        sel = contains(data.labels,tmp);
        data.X = data.X(:,sel);
        data.labels_new = data.labels(sel);
        newlab = data.labels_new;
        prefix = 'spine';
    else
        error('%s not reognized input',inputType)
    end

    data.labels_new = newlab;

function x = perpendicularity(a,b)
    a = a./vecnorm(a,2,2);
    b = b./vecnorm(b,2,2);
    x = vecnorm(cross(a,b,2),2,2);
    
function ds = speed(X,time,labels)
    labels2 = cellfun(@(x) x(1:end-2),labels,'un',0);
    
    [ulab,~,ilab] = unique(labels2);
    nx = numel(labels) ./ numel(ulab);
    
    dt = diff(time);
    d = [zeros(1,size(X,2)); diff(X)./dt];
    
    ds = nan(size(X,1),numel(labels)/nx);
    for ic=1:numel(ulab)
        sel = ilab==ic;
        tmp = sqrt(sum(d(:,sel).^2,2));
        ds(:,ic) = tmp;
    end

    
function v = velocity(X,time)

    dt = diff(time);
    v = [zeros(1,size(X,2)); diff(X)./dt];


function [X,newlab] = segment_length(dat,labels,joint_graph)

X = [];
for ii=1:size(joint_graph,1)
    j1 = joint_graph{ii,1};
    j2 = joint_graph{ii,2};
    
    v1 = dat(:,contains(labels,j1));
    v2 = dat(:,contains(labels,j2));
    
    d = sqrt(sum((v2-v1).^2,2));
    X = cat(2,X,d);
    
    newlab{ii} = sprintf('%s-%s',j1,j2);
end

foo=1;


function [X,newlab] = relative_length(dat,labels,joint_graph)

[d,lab] = segment_length(dat,labels,joint_graph);

cmb = combnk(1:size(d,2),2);

X = nan(size(dat,1),size(cmb,1));
newlab = {};
for ic=1:size(cmb,1)
    ii1 = cmb(ic,1);
    ii2 = cmb(ic,2);
    
    X(:,ic) = d(:,ii1) - d(:,ii2);
    newlab{ic} = sprintf('%s:%s',lab{ii1},lab{ii2});
end

foo=1;

function [X,newlab] = relative_angle(dat,labels,joint_graph)

X = [];
for ii=1:size(joint_graph,1)
    j1 = joint_graph{ii,1};
    j2 = joint_graph{ii,2};
    
    v1 = dat(:,contains(labels,j1));
    v2 = dat(:,contains(labels,j2));
    
    a = atan2(vecnorm(cross(v1,v2,2),2,2),dot(v1,v2,2)); % angle

    X = cat(2,X,a);
    
    newlab{ii} = sprintf('%s-%s',j1,j2);
end

foo=1;


function [X,pairsLabels] = limb_metrics(dat,labels,limbs,metric)

    X = [];
    pairsLabels = {};

    for ii=1:size(limbs,1)
        p = limbs(ii,:);

        % get joints
        p1 = dat(:,contains(labels,p{1}));
        p2 = dat(:,contains(labels,p{2}));
        p3 = dat(:,contains(labels,p{3}));

        % limbs
        v1 = p1 - p2;
        v2 = p3 - p2;

        % metrics
        L1 = sum(v1.^2,2);
        L2 = sum(v2.^2,2);
        
        a = atan2(vecnorm(cross(v1,v2,2),2,2),dot(v1,v2,2)); % angle
        a2 = nan(size(v1)); % angle on each plane
        for ia=1:3
            tmp1 = v1; tmp2 = v2;
            tmp1(:,ia) = 0;
            tmp2(:,ia) = 0;
            a2(:,ia) = atan2(vecnorm(cross(tmp1,tmp2,2),2,2),dot(tmp1,tmp2,2)); % angle
        end

        ds = p1 - p3; %displacement
        d = sqrt(sum((p1-p3).^2,2)); % distance
        ar = sqrt(sum(cross(v1,v2).^2,2))./2; % polygon area
        %h = ar.*2 ./ sqrt(sum((v1-v2).^2,2)); % 
        pr = L1 + L2 + sqrt(sum((v1-v2).^2,2)); % perimeter
        
        % append
        switch metric
            case 'angle'; X = cat(2,X,a);
            case 'angle2'; X = cat(2,X,a2);
            case 'displacement'; X = cat(2,X,ds);
            case 'distance'; X = cat(2,X,d);
            case 'length'; X = cat(2,X,sqrt(L1+L2));
            case 'polyarea'; X = cat(2,X,ar);
            case 'perimeter'; X = cat(2,X,pr);
            otherwise
                error('dont recognize limb metric: %s',metric)
        end

        s = sprintf('%s-%s:%s-%s',p{1},p{2},p{2},p{3});
        pairsLabels{ii} = s;
    end
    foo=1;

    
function [D,new_labels] = hand_metrics(X,labels,m)
    [~,joint_graph] = limb_graph3();
    
    switch m
        case 'distance'
                D = [];
                for ii=1:size(joint_graph,1)
                    j1 = joint_graph{ii,1};
                    j2 = joint_graph{ii,2};
                    p1 = X(:,contains(labels,j1));
                    p2 = X(:,contains(labels,j2));

                    d = sqrt(sum((p1-p2).^2,2));
                    D(:,ii) = d;
                end

                new_labels = cellfun(@(x,y) [x '-' y],joint_graph(:,1),joint_graph(:,2),'un',0);
        case 'area'
            ujoint = {'rhand','rfoot','lfoot','lhand'};
            
            x = [];
            y = [];
            for ii=1:numel(ujoint)
                x = cat(2,x,X(:,contains(labels,[ujoint{ii} '_x'])));
                y = cat(2,y,X(:,contains(labels,[ujoint{ii} '_z'])));
            end

            D = polyarea(x,y,2);
            new_labels = ujoint;
    end
    
foo=1;
   
% //////////////////////////////////////////////////////////////////
% limb defs
function seg = limb_segment1(joint_graph)


function [limbs,joint_graph] = limb_graph1(njoints)

if njoints==13 % 13 joints
    joint_graph = {'neck','head';
        'neck','rshoulder';
        'neck','lshoulder';
        'neck','hip';
        'head','nose';
        'hip','rknee';
        'hip','lknee';
        'hip','tail';
        'lknee','lfoot';
        'rknee','rfoot';
        'rshoulder','rhand';
        'lshoulder','lhand';
        };
elseif njoints==13 % 15 joints
    joint_graph = {'neck','head';
        'neck','rshoulder';
        'neck','lshoulder';
        'neck','hip';
        'head','nose';
        'hip','rknee';
        'hip','lknee';
        'hip','tail';
        'lknee','lfoot';
        'rknee','rfoot';
        'rshoulder','relbow';
        'lshoulder','lelbow';
        'relbow','rhand';
        'lelbow','lhand'};
elseif njoints==19 %  19 joints
    joint_graph = {'neck','head';
        'neck','rshoulder';
        'neck','lshoulder';
        'neck','hip';
        'head','nose';
        'hip','rknee';
        'hip','lknee';
        'hip','mtail';
        'mtail','tail';
        'lknee','lfoot';
        'rknee','rfoot';
        'rshoulder','relbow';
        'lshoulder','lelbow';
        'relbow','rhand';
        'lelbow','lhand'
        'nose','rear';
        'nose','lear'
        };
else
    error('what do?')
end

limbs = convert_to_limbs(joint_graph);


function [limbs,joint_graph] = limb_graph2()

% joint_graph = {    
%     'neck','hip';
%     'hip','lfoot';
%     'hip', 'rfoot';
%     'neck', 'lhand';
%     'neck','rhand';
%     };

joint_graph = {    
    'neck','hip';
    'hip','rfoot';
    'hip','lfoot';
    %'hip','lknee';
    %'hip', 'rknee';
    %'rknee','rfoot';
    %'lknee','lfoot';
    'neck', 'lhand';
    'neck','rhand';
    };

limbs = convert_to_limbs(joint_graph);


function [limbs,joint_graph] = limb_graph3()

joint_graph = {
    'lhand','rhand';
    'rhand','rfoot';
    'rfoot','lfoot';
    'lfoot','lhand';
    %'lfoot','rhand';
    %'rfoot','lhand';

    };

limbs = convert_to_limbs(joint_graph);

function [limbs,joint_graph] = limb_graph4()

% joint_graph = {
%     'lhand','rhand';
%     'lfoot','rfoot';
%     'rhand','rfoot';
%     'lhand','lfoot';
%     'rhand','lfoot';
%     'lhand','rfoot';
% 
%     'lfoot', 'hip';
%     'rfoot', 'hip';
%     'lhand', 'neck';
%     'rhand', 'neck';
%     };

joint_graph = {
    'lhand','rhand';
    'rhand','rfoot';
    'rfoot','lfoot';
    'lfoot','lhand'

    'neck','hip';
    'hip','lfoot';
    'hip','rfoot';
    'neck','lhand';
    'neck','rhand';
    };

limbs = convert_to_limbs(joint_graph);


function [limbs,joint_graph,X] = limb_graph5(X,labels)

joint_graph = {
    'neck','rhand';
    'neck','rfoot';
    'neck','lfoot';
    'neck','lhand';
    %'lfoot','rhand';
    %'rfoot','lhand';
    };

limbs = convert_to_limbs(joint_graph);

% for this, make sure all data is centered on the neck
neck = X(:,contains(labels,'neck'));
neck = repmat(neck,1,size(X,2)/3);
X = X - neck;


function limbs = convert_to_limbs(joint_graph)

G=graph(joint_graph(:,1),joint_graph(:,2));

links = nan(0,3);
for ib=1:numel(G.Nodes)
    idx = neighbors(G,ib);

    tmp = [];
    for ii=1:numel(idx)
        for jj=1:numel(idx)
            if ii==jj; continue; end
            a = [idx(ii), ib, idx(jj)];
            if ismember(a([3 2 1]),links,'rows')
                continue
            else
                links = cat(1,links,a);
            end
        end
    end
    %links = cat(1,links,tmp);
    foo=1;
end

limbs = table2array(G.Nodes);
limbs = limbs([links(:)]);
limbs = reshape(limbs,size(links));
