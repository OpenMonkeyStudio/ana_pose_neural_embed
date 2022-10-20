%{
 assumes:
- reembed_pose3 has been run
%}

%{
- are neurons tuned to pose?
- are neurons tuned to modules?
- are they tuned to either or both?

- does tuning to pose/module etc depend on area?

- how does tuning change as a function of hierarchy?
- does this depend on area?

- example cell highly tuned to pose?

- neurons predict pose transitions? module transitions?
- how far before?
%}

%% paths
[parentdir,jsondir,pyenvpath,rpath] = set_pose_paths(0);

%anadir = [parentdir '/P_neural_embed/embed_rhesus_jointAngle_vcom_pcom_noAlignPose_pca_v2_test'];
% anadir = [parentdir '/P_neural_embed_slim/embed_rhesus_jointAngle_vcom_pcom_noAlignPose_pca_new'];
% anadir = [parentdir '/P_neural_embed_slim/embed_rhesus_jointAngle_vcom_pcom_noAlignPose_pca_new_test'];
%anadir = [parentdir '/P_neural_embed_slim/embed_rhesus_jointAngle_vcom_pcom_noAlignPose_pca_new_13j'];
%anadir = [parentdir '/P_neural_embed_slim/embed_rhesus_jointAngle_vcom_pcom_noAlignPose_pca_new_13j_rangeNorm'];

% anadir = [parentdir '/P_neural_embed_slim/embed_rhesus_jointAngle_vcom_pcom_noAlignPose_pca_new_13j_again'];
%anadir = [parentdir '/P_neural_embed/embed_rhesus_jointAngle_vcom_pcom_noAlignPose_rangeNorm_lowUMAP_stepSplit'];
%anadir = [parentdir '/P_neural_embed/embed_rhesus_jointAngle_vcom_pcom_noAlignPose_rangeNorm_lowUMAP_stepSplit'];
%anadir = [parentdir '/P_neural_embed/embed_rhesus_jointAngle_vcom_pcom_noAlignPose_rangeNorm_lowUMAP_stepSplit_v2'];

%anadir = [parentdir '/P_neural_embed/embed_rhesus_jointAngle_vcom_pcom_noAlignPose_rangeNorm_lowUMAP_stepSplit_v2'];
%anadir = [parentdir '/P_neural_embed_wo/embed_rhesus_jointAngle_vcom_pcom_noAlignPose_rangeNorm_lowUMAP_stepSplit_v2'];

%anadir = [parentdir '/P_neural_embed/embed_rhesus_muchoFeats_lowUMAP_segSplit'];

%anadir = [parentdir '/P_neural_embed_wo/embed_rhesus_jointAngle_vcom_pcom_noAlignPose_rangeNorm_lowUMAP_stepSplit_v2'];
[spkparentpath,~] = fileparts(anadir); % one level up
 
%% settings
nparallel = 20; %gpuDeviceCount()-1;

% useful
fs_frame = 30;
fs_spk = 1000;

fontSize = 12;

areaOrder = {'PM','SMA','DLPFC','VLPFC','ACC','OFC'};

% data
onlyLoadData = 0;

loadEmbeddingData = 0;
    trainData = 0;
    loadModularity = 1;
    cleanClusters = 0;
    minSmp = 3;
    pSmp = 0.001;

getSDF = 0;
    collapseToMUA = 0; % 0=dont, 1=gross subvidions, 2=minor sibdivisions
    forceAreaUpdate = collapseToMUA==0;
    sdfsuffix = '_residTask';
    useSDF_resid = 1;
    calcSDF = 0;
    saveSDF = 0;
    regressOutVariables = 1;
        theseRegressors = {'evt','com'}; % 'jt'
    resid_type = 2; %1=linear, 2=poisson
    timwin = [-0.1 0.1];
    timwin_regressor = [-0.025 0.025];
    loadAreas = {'ACC','VLPFC','DLPFC','OFC','SMA','PM'};
loadSDF = 0;

% more calculations
getMeanRate = 0;
    collapsePoses = 1;
    winlim = [3 ceil(0.5*30)]; % samples

getDensityMaps = 0;
    getMaps = 1;
        mapsIgnore = 4; % 0, 1, 2,3
        overwriteMaps = 0;
    getSampleSimilarity = 0;
getMapMetrics = 0;
    spreadEnergy = 0;

classifyPose = 0;
    classifierType = 'svc'; % svc, xgb
classifyModularityCuts = 0;
    classifierType_mod = 'svc'; % svc, xgb
    svcKernel = 'linear'; % linear, rbf

classifyPoseNeurons = 0;
classifyPoseNeurons_engage = 0;

% analysis
plotResidResults = 0;
    loadFitResults = 0;

plotClassification_pose = 0;
    loadPoseClassification = 1;
plotCorrAccRate = 0;

plotClassifyMod = 0;
    loadClassifyMod = 1;

plotMapsKL = 0;
    normByRand = 0;
plotMapsKL_modularity = 0;
plotMaps_examples = 0;
plotMapsMean = 0;
plotMapsKLvsRate = 0;
plotMapsSimilarity = 0;
plotMapsDensities = 0;

plotEmbeddedRate_pose = 0;

plotMeanRate = 0;
plotMeanRate_engage = 0;
plotCorrRateResidence = 0;

plotTransitionRSA = 0;
    rsaUseDist = 0;
    getTransitionRSA = 1;

plotCorrNeuralVSbehav = 0;

plotPSTH_pose = 1;

plotNeuralDimVSbehavStab = 0;
    calcStab = 1;
plotNeuralDimVSbehavStab_time = 0;
    calcStabTime = 1;
    calcStabTime_byArea = 0;

plotPoseTuning = 0;
plotModuleTuning = 0;
plotTransitionTuning_pose = 0;
plotTransitionTuning_module = 0;
plotHierarchyTuning = 0;

plotSpecificity_pose = 0;
plotSpecificity_pose_byArea = 0;

plotPeriTransition_module = 0;
plotControlRate_module = 0;
plotControlTuning_module = 0;

% example stuff
plotExampleTuning_module = 0;
exampleTuning = 0;
prepSampleVideo = 0;

% clean up paths
tmpresid = {'noresid','resid'};
tmpstr = {'lin','pois'};
muastr = {'','_mua1','_mua2'};
%sdfpath = [anadir '/sdf_fs30_' resstr{useSDF_resid+1} '_' tmpstr{resid_type}];
sdfpath = sprintf('%s/sdf_fs30_resid_%s_%g-%g%s%s',anadir,tmpstr{resid_type},timwin(1)*1000, timwin(2)*1000,sdfsuffix,muastr{collapseToMUA+1});
if ~exist(sdfpath); mkdir(sdfpath); end
sdfpath_data = [sdfpath '/sdf'];
if ~exist(sdfpath_data); mkdir(sdfpath_data); end
    
clfpath = [sdfpath '/clf_' classifierType '_' tmpresid{useSDF_resid+1}];
if ~exist(clfpath); mkdir(clfpath); end
clfmodpath = [sdfpath '/clf_mod_' classifierType_mod '_' tmpresid{useSDF_resid+1}];
if ~exist(clfmodpath); mkdir(clfmodpath); end

mapdir = [sdfpath '/maps_' tmpresid{useSDF_resid+1}];
if ~exist(mapdir); mkdir(mapdir); end

figdir = [sdfpath '/Figures_' tmpresid{useSDF_resid+1}];
if ~exist(figdir); mkdir(figdir); end


%% load embedding data
if loadEmbeddingData

    if trainData
        fprintf('loading training data...\n')

        % load orig data
        fprintf('\torig...\n')
        %load([anadir '/orig_data.mat']);
        tmp = {'idat' 'com' 'frame' 'labels' 'datasets'};
        load([anadir '/data_train.mat'],tmp{:})

        % load cluster results
        % - assumes has already been cleaned
        fprintf('\tcluster...\n')
        cluster_train_orig = load([anadir '/cluster_train.mat']);

        [C,stateMap] = clean_states(cluster_train_orig.clabels,minSmp,pSmp);
        cluster_train = clean_clusterInfo(cluster_train_orig,stateMap);

        % load modularity results
        if loadModularity
            fprintf('\tmodularity...\n')
            modularity_C = load([anadir '/modularity_train/C_mod_train.mat']);
            C_mod = modularity_C.C_mod;
        else
            C_mod = nan(size(C));
        end
        
        % load embedding data
        fprintf('\tembeded samples...\n')
        umap_train = load([anadir '/umap_train.mat']);
        Y_test = umap_train.embedding_;
    else
        fprintf('loading testing data...\n')

        % load orig data
        fprintf('\torig...\n')
        %load([anadir '/orig_data.mat']);
        tmp = {'idat' 'com' 'frame' 'labels' 'datasets'};
        %load([anadir '/data_test.mat'],tmp{:})
        load([anadir '/info.mat'],tmp{:})
        
        % load cluster results
        % - assumes has already been cleaned
        fprintf('\tcluster...\n')
        cluster_train_orig = load([anadir '/cluster_test.mat']);

        if cleanClusters
            [C,stateMap] = clean_states(cluster_train_orig.clabels,minSmp,pSmp);
            cluster_train = clean_clusterInfo(cluster_train_orig,stateMap);
        else
            C = cluster_train_orig.clabels;
            cluster_train = cluster_train_orig;
            stateMap = [1:max(C), 1:max(C)];
        end
        
        % load modularity results
        if loadModularity
            fprintf('\tmodularity...\n')
            modularity_C = load([anadir '/modularity_test/C_mod_train.mat']);
            C_mod = modularity_C.C_mod;
            load([anadir '/modularity_test/modularity_train.mat'])
        else
            C_mod = nan(size(C));
        end
        
        % load embedding data
        fprintf('\tembeded samples...\n')
        umap_train = load([anadir '/umap_test.mat']);
        Y_test = umap_train.embedding_;
    end

    % stuff
    nstate = max(C);

    % map modularity
    if loadModularity
        fprintf('\t mapping pose to modularity...\n')
        pose2mod=zeros(max(idat),nstate);
        for id=1:max(idat)
            for im=1:nstate
                sel = idat==id & C==im;
                c = C_mod(sel);
                if ~isempty(c)
                    pose2mod(id,im) = c(1);
                end
            end
        end
    else
        pose2mod = [];
    end
    
    % cull, for now
    if 0
    badData = 2;

    datasets(badData) = [];
    bad = idat==badData;
    Y_test(bad,:) = [];
    cluster_train_orig.clabels(bad,:) = [];
    com(bad,:) = [];
    frame(bad,:) = [];
    idat(bad,:) = [];
    C_mod(bad) = [];
    end


end


%% get rate for all cells
if getSDF
    startTime = tic;
    if calcSDF
        ignore_warnings = {'MATLAB:singularMatrix','stats:LinearModel:RankDefDesignMat'};
        for iw=1:numel(ignore_warnings); warning('off',ignore_warnings{iw}); end
        
        udat = unique(idat);
        
        % start parallel pool
        if nparallel > 1 && isempty(gcp('nocreate'))
            myPool = parpool('local',nparallel);
        end

        % prevent broadcasting
        frame_tmp = {};
        for id=1:numel(udat)
            frame_tmp{id} = frame(idat==udat(id));
        end

        % call
        startTime = tic;
        parfor id=1:numel(udat)
        %for id=1:numel(udat)
            name = datasets(id).name;
            name = name(1:end-9);
            fprintf('%g: %s\n',id,name)

            try
                % figure out the frames for this dataset
                thisFrame = frame_tmp{id};

                % loop over channels
                spkfolder = [spkparentpath '/' name];
                dcells = dir([spkfolder '/spk*mat']);
                
                % collapse to mua?
                if collapseToMUA>0
                    fprintf('\t collapsing to MUA, type=%g...',collapseToMUA)
                    dcells2 = [];
                    % figure out areas
                    ii = strfind(name,'_');
                    day = datestr( name(ii(1)+1:ii(2)-1) );
                    monk = name(1:2);

                    ch = [];
                    for ic=1:numel(dcells)

                        % stuff
                        chan = dcells(ic).name;
                        ii = strfind(chan,'_');
                        ch(ic) = str2double(chan(ii(end)+3:end-7));
                    end

                    [a,~] = get_area(ch,day,monk,0);
                    if collapseToMUA==1 % use gross subdivisons
                        a = collapse_areas(a);
                    end
                    [ua,~,iar] = unique(a);
                    
                    % collapse over areas
                    for ia=1:numel(ua)
                        fprintf('%s,',ua{ia})
                        seldat = find(iar==ia);
                        
                        % combine all spikes
                        spk = [];
                        for ic=1:numel(seldat)
                            tmp = load([spkfolder '/' dcells(seldat(ic)).name]);
                            if numel(spk)==0
                                spk = tmp.spk;
                            else
                                spk.time{1} = cat(1,spk.time{1},tmp.spk.time{1});
                                spk.trial{1} = cat(1,spk.trial{1},tmp.spk.trial{1});
                                spk.timestamp{1} = cat(1,spk.timestamp{1},tmp.spk.timestamp{1});
                                spk.waveform{1} = cat(3,spk.waveform{1},tmp.spk.waveform{1});
                            end
                            
                        end
                        
                        % update
                        [~,is] = sort(spk.time{1});
                        spk.time{1} = spk.time{1}(is);
                        spk.trial{1} = spk.trial{1}(is);
                        spk.timestamp{1} = spk.timestamp{1}(is);
                        spk.waveform{1} = spk.waveform{1}(:,:,is);

                        spk.label = {'mu'};
                        spk.area = ua(ia);
                        spk.n_neurons = numel(seldat);
                        
                        % save
                        is = strfind(dcells(1).name,'_');
                        sname = dcells(1).name(5:is(5)-1);
                        sname = sprintf('%s/mu%g_%s_nt%gch1.mat',spkfolder,collapseToMUA,sname,ia);
                        tmp=[];
                        tmp.spk = spk;
                        parsave(sname,tmp)
                        
                        [~,tmp]=fileparts(sname);
                        dcells2(ia).name = [tmp '.mat'];
                    end
                    fprintf('\n')
                    
                    % update
                    dcells = dcells2';
                end
                
                for ichan=1:numel(dcells)
                    tic
                    % load
                    chan = dcells(ichan).name;
                    tmp = load([spkfolder '/' chan]);
                    spk = tmp.spk;

                    % stuff
                    ii = strfind(chan,'_');
                    ch = str2double(chan(ii(end)+3:end-7));

                    ii = strfind(name,'_');
                    day = datestr( name(ii(1)+1:ii(2)-1) );
                    monk = name(1:2);
                    
                    % convert to sec
                    for ic=1:numel(spk.time)
                        spk.time{ic} = spk.time{ic} ./ fs_spk;
                    end
                    spk.trialtime = spk.trialtime ./ fs_spk;

                    % get spike density
                    tlim = [0 max(thisFrame)]./fs_frame; %sec
                    %tlim = tlim * fs_spk;

                    if 1 %sdf
                        cfg = [];
                        %cfg.timwin = [-5 5];
                        cfg.timwin = timwin;
                        cfg.latency = tlim;
                        cfg.fsample = fs_frame;
                        sdf = ft_spikedensity(cfg,spk);
                    elseif 0 % psth
                        pcfg = [];
                        pcfg.binsize = diff(timwin);
                        pcfg.latency = tlim;
                        [psth] = ft_spike_psth(pcfg, spk);
                    elseif 0 % manual testing
                        time = spk.trialtime:1/fs_spk:spk.trialtime(2);
                        win = ceil(diff(timwin)*fs_spk);
                        win = ones(1,win) ./ win;
                        sp = spk.time{1};
                        
                        time2 = floor(time*fs_spk);
                        sp2 = floor(sp*fs_spk);
                        tmp = zeros(size(time2));
                        tmp(ismember(time2,sp2)) = 1;
                        
                        S = conv(tmp,win,'same');
                        N = conv(ones(size(tmp)),win,'same');
                        A = S ./ N;
                        %trialPsth = histc(spikeTimes(:), bins); % we deselect the last bin per default
                    end
                    
                    % presave
                    tmpname = [sdfpath_data '/' name '_ch' num2str(ch) '_sdf.mat'];
                    parsave(tmpname,sdf)

                    % regress out stuff
                    if regressOutVariables
                        fprintf('\t regressing out relevant variables...\n')

                        regpath = [spkfolder '/regressors.mat'];

                        out_reg = load(regpath);
                        reg_info = out_reg.info;
                        out_reg = rmfield(out_reg,'info');

                        % prep regressors
                        labels_reg = fields(out_reg);
                        selr = contains(labels_reg,theseRegressors);
                        labels_reg(~selr) = [];
                        
                        r = zeros(numel(out_reg.(labels_reg{1})),numel(labels_reg));
                        for ireg=1:numel(labels_reg)
                            r(:,ireg) = out_reg.(labels_reg{ireg});
                        end

                        % smooth out regressors
                        thisFS = fs_frame;
                        s = ceil(diff(timwin_regressor) .* thisFS);
                        win = gausswin(s);
                        win = win ./ sum(win);

                        tmp = convn(r,win,'same');
                        n = convn(ones(size(r)),win,'same');
                        tmp = tmp ./ n;
                        r = tmp;

                        % match times to pose
                        t = reg_info.time;
                        selt = find( t >= sdf.time(1) & t <= sdf.time(end) );
                        if numel(selt) < numel(sdf.time)
                            selt(end+1) = selt(end)+1;
                        end
                        r = r(selt,:);
                        
                        % fit a model
                        sdf.avg_resid = sdf.avg;
                        sdf.warn = {};
                        for ic=1:size(sdf.avg,1)
                            if resid_type==1
                                mdl = fitglm(r,sdf.avg(ic,:));
                            elseif resid_type==2
                                mdl = fitglm(r,sdf.avg(ic,:),'Distribution','poisson','link','log');
                            else
                                error('unknown resid type: %g',resid_type)
                            end
                            [LASTMSG, msgid] = lastwarn;
                            lastwarn('')
                            
                            % residualize
                            y_pred = predict(mdl,r)';
                            y_res = sdf.avg(ic,:) - y_pred;

                            sdf.avg_resid(ic,:) = y_res;
                            sdf.warn{ic} = LASTMSG;
                            
                            % save model stuff
                            if 1
                                fprintf('\t saving fit model...\n')
                                tmpname = [sdfpath_data '/' name '_ch' num2str(ch) '_' sdf.label{ic} '_regmdl.mat'];

                                tmpmdl = get_mdl_estimates(mdl);
                                tmpmdl.labels_reg = labels_reg;
                                parsave(tmpname,tmpmdl)
                            end
                            mdl=[];
                        end
                    end
                    
                    % downsample to match capture
                    tq = thisFrame;
                    vq = [];
                    vq_resid = [];
                    for ic=1:size(sdf.avg,1)
                        t = sdf.time;
                        t = t * fs_frame;
                        v = sdf.avg(ic,:);
                        vq(ic,:) = interp1(t,v,tq ,'pchip',nan);

                        if regressOutVariables
                            vr = sdf.avg_resid(ic,:);
                            vq_resid(ic,:) = interp1(t,vr,tq ,'pchip',nan);
                        end
                    end

                    % get area
                    if collapseToMUA>0
                        a = spk.area;
                    else
                        if 1
                            fprintf('\t getting area...\n')
                            [a,~] = get_area(ch,day,monk,0);
                        else
                            a = '!';
                        end
                    end
                    
                    % store
                    for ic=1:size(vq,1)
                        tmp = [];
                        tmp.sdf = vq(ic,:);
                        tmp.frame = tq;
                        tmp.label = sdf.label(ic);
                        tmp.area = a;
                        tmp.ch = ch;
                        tmp.day = day;
                        tmp.id = udat(id);
                        tmp.warn = sdf.warn;

                        if regressOutVariables
                            tmp.sdf_resid = vq_resid(ic,:);
                        end
                        if collapseToMUA
                            tmp.n_neurons = spk.n_neurons;
                        end
                        
                        tmpname = [sdfpath_data '/' name '_ch' num2str(ch) '_' sdf.label{ic} '_sdf_matched.mat'];
                        parsave(tmpname,tmp)
                        %SDF = cat(1,SDF,tmp);
                    end
                    
                    toc
                end
            catch err
                try
                    tmpname = [sdfpath_data '/err_' name '_ch' num2str(ch) '_' tmp.label{ic} '.mat'];
                catch
                    tmpname = [sdfpath_data '/err_' name '.mat'];
                end
                
                tmperr = [];
                tmperr.name = name;
                tmperr.err = err;
                tmperr.id = id;
                try, tmperr.ch = ch; end
                try, tmperr.ic = ic; end
                parsave(tmpname,tmperr)
            end
        end
        
        fprintf('\n\n TOTAL SDF calc: %g\n', toc(startTime))     
                
        for iw=1:numel(ignore_warnings); warning('on',ignore_warnings{iw}); end

        delete(gcp('nocreate'))
    end
    
    % load everything back
    [SDF,sdfnames] = load_back_sdf(sdfpath_data,useSDF_resid,{},{},loadAreas,1,forceAreaUpdate);

    % save orig
    if saveSDF
        fprintf('saving full SDF. Resid=%g %s...\n',useSDF_resid)
        if useSDF_resid
            sname = [sdfpath '/SDF_resid.mat'];
        else
            sname = [sdfpath '/SDF_orig.mat'];
        end
        save(sname,'SDF','-v7.3')
    end
    
    toc(startTime)
elseif loadSDF
    fprintf('re-loading all SDF...\n')
    if useSDF_resid
        sname = [sdfpath '/SDF_resid.mat'];
    else
        sname = [sdfpath '/SDF_orig.mat'];
    end
    load(sname)
end

% collapse areas for analysis
if 1
    areas = [SDF.area]';
    areas = collapse_areas(areas);
    [uarea,~,iarea] = unique(areas);
    
    % remap for plotting
    [~,iarea2] = ismember(areas,areaOrder);
    tmp = [iarea, iarea2];
    tmp = unique(tmp,'rows');
    iarea = changem(iarea,tmp(:,2),tmp(:,1));
    uarea = areaOrder;
    
end


%% /////////////////////////////////////////////////////////
% //////////////////////////////////////////////////////////
if onlyLoadData
    return
end
% //////////////////////////////////////////////////////////
% //////////////////////////////////////////////////////////


%% get mean rate per pose
if getMeanRate
    sw = 30*60*10; % samples
    comparePrc = [0 50 100];
    minEngageSamples = 100;
    
    fprintf('getting mean rates')
    RES_rate = [];
    for is=1:numel(SDF)  
        dotdotdot(is,0.1,numel(SDF))
        day = datestr(SDF(is).day,'yyyy-mm-dd');

        % get data
        datname = SDF(is).name(1:23);
        datfold = [spkparentpath  '/' datname];
        [eng,f_eng] = get_task_engagement(datfold,sw);

        id=SDF(is).id;

        ff = frame(idat==id);
        c = C(idat==id);
        s = SDF(is).sdf;
        fs = SDF(is).frame;
        
        % match data
        [good_frames1,ia,ib] = intersect(fs,f_eng);
        [good_frames2,ix,iy] = intersect(good_frames1,ff);
        ia=ia(ix);
        ib=ib(ix);

        s = s(ia);
        c = c(iy);
        ff = ff(iy);
        eng = eng(ib);
        if isrow(s); s=s'; end

        % collapse poses?
        if collapsePoses
            [s,c,istate] = collapse_poses(s,c,winlim);
            [eng,~,istate] = collapse_poses(eng,[],winlim,istate);
        end
        
        % split engagment
        nsplit = numel(comparePrc);
        noEngage = eng==0;

        % doesnt engage enough
        if (numel(noEngage) - sum(noEngage)) < ...
                minEngageSamples*numel(comparePrc)
            ieng = ones(size(noEngage));
        else
            prc = prctile(eng(~noEngage),comparePrc);
            [~,~,ieng] = histcounts(eng,prc);
            ieng=ieng+1;

            % if too few samples, just collapse it all together
            if sum(ieng<=1) < minEngageSamples
                ieng(ieng<=1) = 2;
            end
        end
        ieng = ieng';
        
        % get mean rate 
        mu=nan(nstate,nsplit+1);
        se=nan(nstate,nsplit+1);
        n = mu;
        for ic=1:nstate
            for ig=1:nsplit+1
                if ig==1
                    selc = c==ic;
                else
                    selc = c==ic & ieng==ig-1;
                end
                tmp = s(selc);
                mu(ic,ig) = nanmean(tmp);
                se(ic,ig) = nanstd(tmp);
                n(ic,ig) = sum(selc);
            end
        end
        
        % store
        tmp = [];
        tmp.name = SDF(is).name;
        tmp.rate_pose = mu;
        tmp.rate_pose_sd = se;
        tmp.dataset = datasets(id).name;
        tmp.pose_residence = n;
        RES_rate = cat(1,RES_rate,tmp);

    end
end
  


%% run pose claissification
if classifyPose

    maxWin = ceil(1*30); % samples
    minWin = 3;
    
    if nparallel > 1 && isempty(gcp('nocreate'))
        if strcmp(classifierType,'xgb') % uses gpu, so enforce gpu number
            ngpu = gpuDeviceCount()-1;
            myPool = parpool(ngpu);
        else
            myPool = parpool(nparallel);
        end
    end
    
    % prep to prevent broadcast
    C_noBroadcast = {};
    for id=1:numel(datasets)
        C_noBroadcast{id} = C(idat==id);
    end
    
    % loop over datasets
    T_start = tic;
    parfor id=1:numel(datasets)
    %for id=20%1:numel(datasets)

        dotdotdot(id,0.1,numel(datasets));

        name = datasets(id).name(1:end-9);
        day = datestr( name(4:end-10) );
        fprintf('%g: %s\n',id,name)
               
        % check if the processing went through
        sname = [clfpath '/' name '_clf.mat'];
        if exist(sname,'file')
            tmpin = load(sname);
            if isfield(tmpin,'coef_obs')
                continue
            end
        end
        
        % select
        if nparallel > 1 % faster to load it back for the day
            sdf = load_back_sdf(sdfpath_data,useSDF_resid,{day},{},loadAreas,1,forceAreaUpdate);
        else
            selsdf = ismember({SDF.day},day);
            sdf = SDF(selsdf);
        end

        if numel(sdf)<2
            continue
        end

        % concatenate all datasets
        uframe = unique([sdf(:).frame]);

        X = nan(numel(uframe),numel(sdf));
        for is=1:numel(sdf)
            x = sdf(is).sdf;

            % reduce outlier influence
            bad = abs(x) > 50;
            idx=1:numel(x);
            xi = interp1(idx(~bad),x(~bad),find(bad),'pchip','extrap');
            x(bad) = xi;
            %x = medfilt1(x,15);

            % store
            [~,self] = ismember(uframe,sdf(is).frame);
            X(self,is) = x;
        end

        % normalize        
        %X = X ./ max(X);
        X = (X-nanmean(X)) ./ nanstd(X);
        
        % split into "trials"
        c = C_noBroadcast{id};
        [Xt,y,istate] = collapse_poses(X,c,[minWin maxWin]);
        
        % call python
        dat=[];
        dat.X = Xt;
        dat.y = y;
        dat.nrand = 20;
        dat.nfold = 5;
        
        opts = [];
        opts.pyenvpath = pyenvpath;
        opts.verbose = 0;
        opts.clean_files = 1;
        opts.add_orig_data = 0;

        if strcmp(classifierType,'svc')
            opts.func = [get_code_path() '/bhv_cluster/matlab/python/call_svc.py'];
        elseif strcmp(classifierType,'xgb')
            opts.func = [get_code_path() '/bhv_cluster/matlab/python/call_xgb.py'];
            dat.gpu_id = mod(id-1,ngpu);
        end
            
        tic
        out = sendToPython(dat,opts);
        toc
        
        % append soome more info
        out.area = [sdf.area];
        out.id = id;
        out.name = datasets(id).name;
        
        % save
        parsave(sname,out)
        %clear out
    end
    delete(gcp('nocreate'))
    toc(T_start)
end
   

%% classificaiton on individual neurons
if classifyPoseNeurons
    getAnova = 0;
        testtype = 'kw';
        nrand = 20;
        nboot = 1;
        smoothWin = round( 0.5*30*fs );
        ilag = 1;

    avgtype = 'mean';
    useRand = 0;
    
    if getAnova
        maxWin = ceil(1*30); % samples
        minWin = 3;

        %theseCuts = [Cutoffs(1:5), Cutoffs(6:2:15), Cutoffs(10:5:end), nstate];
        theseCuts = [2:8 10:2:20 23:3:31, nstate];
        %theseCuts = 40:50;
        
        ncuts = numel(theseCuts);

        % loop over cells
        fprintf('indiv classification...')
        tic

        % prep to prevent broadcast
        C_noBroadcast = {};
        for id=1:numel(datasets)
            C_noBroadcast{id} = C(idat==id);
        end
        CHAN = [SDF.ch];

        F = nan(numel(SDF),numel(theseCuts));
        Fr = nan(numel(SDF),numel(theseCuts),nrand);
        P = nan(numel(SDF),numel(theseCuts));

        nrand2 = nrand+1;
        parfor id=1:numel(SDF)
        %for id=1:numel(SDF)
            %try
                disp(id)
                sdf = SDF(id);

                % get data
                uframe = unique([sdf(:).frame]);
                X = sdf.sdf;
                id2 = sdf.id;        
              
                % prep cuts
                tmp = squeeze(Qh_lab_all(ilag,id2,:,:));
                m = max(tmp,[],1);
                maxCuts = max(m);

                dat_cuts = cell(numel(theseCuts),3);
                for ic1=1:numel(theseCuts)
                    cut = theseCuts(ic1);
                    if cut==nstate
                        c = C_noBroadcast{id2};
                    else
                        if cut > maxCuts
                            c = [];
                            continue
                        else
                            % remap
                            ic = ismember(Cutoffs,cut);

                            newval = squeeze(Qh_lab_all(ilag,id2,:,ic))+1;
                            oldval = 1:nstate;
                            c = C_noBroadcast{id2};
                            c = changem(c,newval,oldval);
                            c = clean_states(c,3,0);
                        end
                    end
                    
                    % split into "trials"
                    [Xt,y,istate] = collapse_poses(X,c,[minWin maxWin]);

                    % store
                    dat_cuts{ic1,1} = Xt';
                    dat_cuts{ic1,2} = y;
                    dat_cuts{ic1,3} = c;
                    
                end
                ndat = cellfun(@numel,dat_cuts(:,2));
                mndat = min(ndat);
                
                % loopp over all cuts
                for ic1=1:ncuts
                    fprintf('%g,',ic1)
                    
                    for ib=1:nboot
                        Xt = dat_cuts{ic1,1};
                        y = dat_cuts{ic1,2};
                        
                        if isempty(Xt); continue; end
                        
                        if nboot > 1
                            bidx = randperm(numel(Xt),mndat);
                            bidx = sort(bidx);
                            Xt = Xt(bidx);
                            y = y(bidx);
                        end
                        
                        % loop calculations
                        for ir=1:nrand2
                            % rand or no?
                            if ir <= nrand
                                ridx = randi( ceil([0.25* 0.75]*numel(y)) );
                                y2 = circshift(y,ridx);
                            else
                                y2 = y;
                            end

                            % run anova
                            if strcmp(testtype,'anova')
                                [~,T,~,~] = anovan(Xt,y2,'display','off');
                                f = T{2,6};
                                p = T{2,7};
                            else
                                [~,T,~] = kruskalwallis(Xt,y2,'off');
                                f = T{2,5};
                                p = T{2,6};
                            end

                            % store
                            if ir==nrand2
                                F(id,ic1,ib) = f;
                                P(id,ic1,ib) = p;
                            else
                                Fr(id,ic1,ib,ir) = f;
                            end
                        end
                    end
                end
                fprintf('\n')
    %         catch err
    %             disp(err.message(end))
    %             error('error: %g',id)
    %         end

        end
            
        % save for later
        fprintf('saving...\n')
        sname = [sdfpath '/' testtype '_actions.mat'];
        save(sname,'F','P','Fr','theseCuts')
        
        toc
    end
    
    % prep
    if ~useRand
        A = nanmean(F,3);
        P2 = P;
    else
        A = nanmean(F - nanmean(Fr,4),3);
        P2 = sum(Fr > F,3) ./ nrand;
    end
    
    % get norms by session
    tmpa = nan([size(A) 2]);
    tmpa(:,:,1) = A;
    for id=1:numel(datasets)
        sel = [SDF.id]==id;
        
        tmp = A(sel,:);
        %tmp = tmp ./ max(tmp(:));
        tmp = tmp - nanmean(tmp);
        tmpa(sel,:,2) = tmp;
    end
    A = tmpa;
    
    % get engagement
    eng = [];
    for id=1:numel(datasets)
        datname = datasets(id).name(1:end-9);
        datfold = [spkparentpath  '/' datname];
        [eng(id),f_eng] = get_task_engagement(datfold,0);
    end
    eng = eng([SDF.id]);
    
    % finish prep
    [G,~,iG] = unique(iarea);
    
    xx = theseCuts(1:end-1);
    A_act = A(:,end,:);
    A = A(:,1:end-1,:);

    % ------------------------------------------------
    % plot
    
    %xx(end-1) = [];
    
    figure;
    nr = 2; nc = 4;
    set_bigfig(gcf,[0.4 0.35],[0.4 0.27])
    set(gcf,'units','normalized')
    pos = get(gcf,'position');
    
    cols = get_safe_colors(0,[1:5 7]);
    tstrs = {'raw F','session-norm F'};
    
    % plot grand mean
    [mu,se] = avganderror(A(:,:,1),avgtype,1,1,200);
    %[mur,ser] = avganderror(Ar,avgtype);

    subplot(nr,nc,1)
    h = [];
    htmp = shadedErrorBar(xx,mu,se,{'r-'});
    h(1) = htmp.mainLine;
    hold all
    %htmp = shadedErrorBar(xx,mur,ser,{'k-'});
    %h(2) = htmp.mainLine;

    title('decoding performance per session')
    xlabel('nclust')
    ylabel([avgtype ' F' ])
    %legend(h,{'obs','rand'})
    axis square
    
    for ip=1:2  
        % ---------------------------------------
        % split by area
        [mu,se] = avganderror_group(iG,A(:,:,ip),avgtype,200);
        %[mur,ser] = avganderror_group(iG,coefr,avgtype);

        ns = 2 + nc*(ip-1);
        subplot(nr,nc,ns)
        %hp = plot(xx,mu','.-');
        for ii=1:size(mu,1)
            col = cols(ii,:);
            htmp = shadedErrorBar(xx,mu(ii,:),se(ii,:),{'.-','color',col});
            hold all
            hp(ii) = htmp.mainLine;
        end
        %hold all
        %plot(xx,mur','--');

        s = sprintf('influence of each area on decoding\n%s',tstrs{ip});
        title(s)
        xlabel('nclust')
        ylabel([avgtype ' F'])
        legend(hp,uarea,'location','northwest')
        axis square
    
        % ---------------------------------------
        % plot decoding for lowest level 
        [mu,se] = avganderror_group(iG,A_act(:,:,ip),avgtype,200);
        [~,tmp,~,~] = anovan(A_act(:,:,ip),iG,'display','off');
        p = tmp{2,7};
        t = tmp{2,6};
        
        ns = 3 + nc*(ip-1);
        subplot(nr,nc,ns)
        for ii=1:size(mu,1)
            m = mu(ii);
            s = se(ii);
            hb = barwitherr(s,ii,m);
            set(hb,'facecolor',cols(ii,:))
            hold all
        end

        s = sprintf('encoding actions\n anovan F=%.3g, p=%.3g',t,p);
        title(s)
        ylabel([avgtype ' F'])

        set(gca,'xtick',1:numel(mu),'xticklabel',uarea)
        axis square
        
        % ---------------------------------------
        % correlate with engagement
        [r,p,ugroup] = corr_group(iG,eng',A_act,'type','spearman');
        
        ns = 4 + nc*(ip-1);
        subplot(nr,nc,ns)
        for ii=1:size(mu,1)
            m = r(ii);
            s = nan;
            hb = barwitherr(s,ii,m);
            set(hb,'facecolor',cols(ii,:))
            hold all
        end
        
        title('correlate with engagment')
        ylabel(['spearman R'])

        set(gca,'xtick',1:numel(mu),'xticklabel',uarea)
        axis square
    end
    
        
    
    % save
    sname = [figdir '/decoding_' testtype '.pdf'];
    save2pdf(sname,gcf)

end




%% classificaiton on individual neurons
if classifyPoseNeurons_engage
    getAnova = 1;
        testtype = 'kw';
        nrand = 0;
        nboot = 50;
        smoothWin = round( 1*60*fs );
        prc = [0 50 100];

    avgtype = 'median';
    useRand = 0;
    
    if getAnova
        maxWin = ceil(1*30); % samples
        minWin = 3;

        %theseCuts = [Cutoffs(1:5), Cutoffs(6:2:15), Cutoffs(10:5:end), nstate];
        theseCuts = [2:8 10:2:20 23:3:31, nstate];
        %theseCuts = 40:50;
        
        ncuts = numel(theseCuts);

        % loop over cells
        fprintf('indiv classification...')
        tic

        % prep to prevent broadcast
        C_noBroadcast = {};
        for id=1:numel(datasets)
            C_noBroadcast{id,1} = C(idat==id);
            C_noBroadcast{id,2} = frame(idat==id);
        end
        CHAN = [SDF.ch];

        F = nan(numel(SDF),numel(prc),nboot);
        Fr = nan(numel(SDF),numel(prc),nboot,nrand);
        P = nan(numel(SDF),numel(prc),nboot);

        nrand2 = nrand+1;
        nprc = numel(prc);
        parfor id=1:numel(SDF)
        %for id=1%1:numel(SDF)
            %try
                disp(id)
                sdf = SDF(id);

                % get data
                uframe = unique([sdf(:).frame]);
                X = sdf.sdf;
                id2 = sdf.id;        
                datname = sdf.name(1:23);
                
                % task engagement
                datfold = [spkparentpath  '/' datname];
                [eng,f_eng] = get_task_engagement(datfold,smoothWin);

                % match data
                [good_frames1,ia,ib] = intersect(sdf.frame,f_eng);
                [good_frames2,ix,iy] = intersect(good_frames1,C_noBroadcast{id2,2});
                ia=ia(ix);
                ib=ib(ix);

                bad = find(isnan(X));
                bad = ismember(ib,bad);
                ia(bad) = [];
                ib(bad) = [];
                iy(bad) = [];

                f2 = C_noBroadcast{id2,2}(iy);
                c = C_noBroadcast{id2,1}(iy);
                X = X(ia)';
                eng = eng(ib);
        
                % collapse poses
                [Xt,y,istate] = collapse_poses(X,c,[minWin maxWin]);
                [eng,~,istate] = collapse_poses(eng,[],winlim,istate);
        
                eng2 = eng(eng>0);
                pp = prctile(eng2,prc);
                if numel(eng2) < 10
                    continue
                end
                [neng,~,ieng] = histcounts(eng',[0 pp]);
                
                minBoot = min(neng);
                
                % loop over engagments
                for ip=1:nprc
                    seleng = find( ieng==ip );
                    if numel(seleng) < 10
                        continue
                    end
                    
                    % control for diff numbers of samples
                    for iboot=1:nboot
                        bidx = randperm(neng(ip),minBoot);

                        Xt2 = Xt(seleng(bidx));
                        y2 = y(seleng(bidx));

                        for ir=1:nrand2
                            % rand or no?
                            if ir <= nrand
                                ridx = randi( ceil([0.25* 0.75]*numel(y2)) );
                                y3 = circshift(y2,ridx);
                            else
                                y3 = y2;
                            end

                            % run anova
                            if strcmp(testtype,'anova')
                                [~,T,~,~] = anovan(Xt2,y3,'display','off');
                                f = T{2,6};
                                p = T{2,7};
                            else
                                [~,T,~] = kruskalwallis(Xt2,y3,'off');
                                f = T{2,5};
                                p = T{2,6};
                            end

                            % store
                            if ir==nrand2
                                F(id,ip,iboot) = f;
                                P(id,ip,iboot) = p;
                            else
                                Fr(id,ip,iboot,ir) = f;
                            end
                        end
                    end
                end
                %fprintf('\n')
    %         catch err
    %             disp(err.message(end))
    %             error('error: %g',id)
    %         end

        end
            
        % save for later
        fprintf('saving...\n')
        sname = sprintf('%s/%s_actions_engage_win%g.mat',sdfpath,testtype,smoothWin);
        save(sname,'F','P','Fr','prc')
        
        toc
    end
    
    % area
    if ~useRand
        A = nanmean(F,3);
        P2 = P;
    else
        A = nanmean(F - nanmean(Fr,4),3);
        P2 = sum(Fr > F,3) ./ nrand;
    end
    
    [G,~,iG] = unique(iarea);
    
    % ------------------------------------------------
    % plot
    xx = prc;
    %xx(end-1) = [];
    
    figure;
    nr = 1; nc = 3;
    set_bigfig(gcf,[0.4 0.25])
    set(gcf,'units','normalized')
    pos = get(gcf,'position');
    set(gcf,'position',[0.4 0.27 pos(3:4)])
    
    % plot accuracy
    [mu,se] = avganderror(A,avgtype,1,1,200);
    %[mur,ser] = avganderror(Ar,avgtype);

    subplot(nr,nc,1)
    h = [];
    htmp = shadedErrorBar(xx,mu,se,{'r-'});
    h(1) = htmp.mainLine;
    hold all
    %htmp = shadedErrorBar(xx,mur,ser,{'k-'});
    %h(2) = htmp.mainLine;

    title('decoding performance per session')
    xlabel('engagment')
    ylabel([avgtype ' excess ' astr])
    %legend(h,{'obs','rand'})
    axis square
    
    % split by area
    %tmpc = coef - coefr;
    tmpc = A;
    [mu,se] = avganderror_group(iG,tmpc,avgtype,200);
    %[mur,ser] = avganderror_group(iG,coefr,avgtype);
    
    subplot(nr,nc,2)
    hp = plot(xx,mu','.-');
    %hold all
    %plot(xx,mur','--');

    title('influence of each area on decoding')
    xlabel('engagment')
    ylabel([avgtype ' F'])
    legend(uarea,'location','southeast')
    axis square
    
    % percent change
    p = (A./A(:,1)-1)*100;
    [mu,se] = avganderror_group(iG,p,avgtype,200);

    subplot(nr,nc,3)
    hp = plot(xx,mu','.-');
    %hold all
    %plot(xx,mur','--');

    title('percent change in encoding of each area')
    xlabel('engagment')
    ylabel([avgtype ' percent change'])
    legend(uarea,'location','southeast')
    axis square
  
end



%% run pose claissification
if classifyModularityCuts

    maxWin = ceil(1*30); % samples
    minWin = 3;
    
    if nparallel > 1 && isempty(gcp('nocreate'))
        if strcmp(classifierType,'xgb') % uses gpu, so enforce gpu number
            ngpu = gpuDeviceCount()-1;
            myPool = parpool(ngpu);
        else
            myPool = parpool(nparallel);
        end
    end
    
    % prep to prevent broadcast
    C_noBroadcast = {};
    for id=1:numel(datasets)
        C_noBroadcast{id} = C(idat==id);
    end
    
    % loop over datasets
    T_start = tic;
    parfor id=1:numel(datasets)
    %for id=70%1:numel(datasets)

        dotdotdot(id,0.1,numel(datasets));

        name = datasets(id).name(1:end-9);
        day = datestr( name(4:end-10) );
        fprintf('%g: %s\n',id,name)
               
        % check if the processing went through
        sname = [clfmodpath '/' name '_clf.mat'];
        if exist(sname,'file')
            tmpin = load(sname);
            if isfield(tmpin,'coef_obs')
                continue
            end
        end
        
        % select
        if nparallel > 1 % faster to load it back for the day
            sdf = load_back_sdf(sdfpath_data,useSDF_resid,{day},{},loadAreas,1,forceAreaUpdate);
        else
            selsdf = ismember({SDF.day},day);
            sdf = SDF(selsdf);
        end

        if numel(sdf)<2
            continue
        end

        % concatenate all datasets
        uframe = unique([sdf(:).frame]);

        X = nan(numel(uframe),numel(sdf));
        for is=1:numel(sdf)
            x = sdf(is).sdf;

            % reduce outlier influence
            bad = abs(x) > 50;
            idx=1:numel(x);
            xi = interp1(idx(~bad),x(~bad),find(bad),'pchip','extrap');
            x(bad) = xi;
            %x = medfilt1(x,15);

            % store
            [~,self] = ismember(uframe,sdf(is).frame);
            X(self,is) = x;
        end

        % normalize        
        %X = X ./ max(X);
        X = (X-nanmean(X)) ./ nanstd(X);
        
        % loop over all cuts
        ilag = 1;
        tmp = squeeze(Qh_lab_all(ilag,id,:,:));
        m = max(tmp,[],1);
        maxCuts = max(m);
        %theseCuts = [Cutoffs(1:5), Cutoffs(6:2:15), Cutoffs(10:5:end), nstate];
        theseCuts = [2:8 10:2:20 25:5:nstate, nstate];

        try
        tic
        out_all = [];
        fprintf('classification on cuts:')
        for ic1=1:numel(theseCuts)
            fprintf('%g,',ic1)

            % extract cluster labels
            cut = theseCuts(ic1);
            if cut==nstate
                c = C_noBroadcast{id};
            else % remap
                % no more cuts?
                if cut > maxCuts
                    tmp = out_all(1);
                    ff = fields(tmp);
                    for ii=1:numel(ff)
                        tmp.(ff{ii}) = nan(size(tmp.(ff{ii})));
                    end
                    out_all = cat(1,out_all,tmp);
                    continue
                end

                % remap
                ic = ismember(Cutoffs,cut);
                
                newval = squeeze(Qh_lab_all(ilag,id,:,ic))+1;
                oldval = 1:nstate;
                c = C_noBroadcast{id};
                c = changem(c,newval,oldval);
                c = clean_states(c,3,0);
            end
            
            % split into "trials"
            [Xt,y,istate] = collapse_poses(X,c,[minWin maxWin]);
           
            % call python
            dat=[];
            dat.X = Xt;
            dat.y = y;
            dat.nrand = 5;
            dat.nfold = 5;
            
            opts = [];
            opts.pyenvpath = pyenvpath;
            opts.verbose = 0;
            opts.clean_files = 1;
            opts.add_orig_data = 0;
    
            if contains(classifierType_mod,'svc')
                opts.func = [get_code_path() '/bhv_cluster/matlab/python/call_svc.py'];
                dat.kernel = svcKernel;
            elseif strcmp(classifierType_mod,'xgb')
                opts.func = [get_code_path() '/bhv_cluster/matlab/python/call_xgb.py'];
                dat.gpu_id = mod(id-1,ngpu);
            else
                error('which classifier?')
            end
               
            %tmpt=tic;
            out = sendToPython(dat,opts);
            %toc(tmpt)

            % store
            out_all = cat(1,out_all,out);
        end
        fprintf('\n')
        toc
        
        % save
        tmp = [];
        tmp.out = out_all;
        tmp.area = [sdf.area];
        tmp.id = id;
        tmp.name = datasets(id).name;
        tmp.Cutoffs = theseCuts;
        
        parsave(sname,tmp)

        catch err
            error('error on %g, %s',id,name)
        end
        %clear out
    end
    delete(gcp('nocreate'))
    toc(T_start)
end


%% correspondence of rate and embedding maps
if getDensityMaps
    % save paths
    mapdir2 = [mapdir '/embedding_maps'];
    if ~exist(mapdir2); mkdir(mapdir2); end
    
    % calculate maps?
    if getMaps
        % settings
        nrand_maps = 25;
        nrand = 15;
        
        if nparallel > 1 && isempty(gcp('nocreate'))
            myPool = parpool('local',nparallel);
        end
        
        % split for parallel processing
        Y_par = {};
        frame_par = {};
        C_par = {};
        for id=1:numel(datasets)
            seldat = idat==id;
            frame_par{id} = frame(seldat);
            Y_par{id} = Y_test(seldat,:);
            C_par{id} = C(seldat);
        end
        
        SDF_par = cell(size(SDF));
        for ii=1:numel(SDF)
            SDF_par{ii} = SDF(ii);
        end
        
        fprintf('rate weighted density maps\n')
        startTime = tic;
        nsdf = numel(SDF);
        parfor ii=1:nsdf
        %for ii=1:nsdf
            tic
            fprintf('map %g density \n',ii)
            name = sdfnames{ii};
            datname = name(1:23);
            sdf_in = SDF_par{ii};
            id = sdf_in.id;

            sname = [mapdir '/' name(1:end-12) '_maps.mat'];

            if overwriteMaps && exist(sname,'file')
                warning('exists, skipping...')
                continue
            end
            
            % spike info
            f1 = sdf_in.frame;
            sdf = sdf_in.sdf;

            % pose info
            [~,ia,ib] = intersect(frame_par{id},f1);
            
            f2 = frame_par{id}(ia);
            y = Y_par{id}(ia,:);
            c = C_par{id}(ia);
            sdf = sdf(ib);

            % ignore samples, for speed
            if mapsIgnore==1
                mu = nanmedian(sdf);
                bad = sdf < mu;
                sdf(bad) = [];
                y(bad,:) = [];
            elseif mapsIgnore==2
                bad = sdf < 0;
                sdf(bad) = 0;
            elseif mapsIgnore==3
                step = 10;
                sdf=sdf(1:step:end);
                y = y(1:step:end,:);
            elseif mapsIgnore==4
                mu = nanmedian(sdf);
                sdf2 = sdf-mu;
                sdf(sdf<0) = 0;
            end

            % re-calc embedding density for this session
            pts = cluster_train.outClust.pts;

            tmpname = [mapdir2 '/' datname '_density.mat'];
            if ~ismember(mapsIgnore,[0 4]) % if ignore for each SDF, recalc
                [dens_embed,~,bw_embed] = ksdens(y,pts);
            elseif ismember(mapsIgnore,[0 4])
                if ~exist(tmpname,'file') % calc fresh
                    [dens_embed,~,bw_embed] = ksdens(y,pts);
                    tmp = [];
                    tmp.dens_embed = dens_embed;
                    tmp.bw_embed = bw_embed;
                    parsave(tmpname,tmp);
                else % use old calc
                    tmp=load(tmpname);
                    dens_embed = tmp.dens_embed;
                    bw_embed = tmp.bw_embed;
                end
            end
            
            % estimate rate-weighted density
            w = sdf - min(sdf);
            w = w ./ max(w);
            
            % speed up by ignoring zeros. NB: have to use same bandwidth!
            bad = w<=0;
            y2 = y(~bad,:);
            w2 = w(~bad);
            [dens_rate,~,bw_rate] = ksdens(y2,pts,bw_embed,'weights',w2);
            %[dens_rate2,~,bw_rate2] = ksdens(y,pts,bw_embed,'weights',w);
            
            % randomization
            if nrand_maps>0
                disp('rand')
                %ko = kldiv_nd(dens_rate,dens_embed);

                kr = [];
                dens_rate_rand = nan([size(dens_rate),nrand_maps]);
                for ir=1:nrand_maps
                    % shift to preserve auto-correlations
                    cn = randi([ceil(numel(w)/4),3*ceil(numel(w)/4)]);
                    wr = circshift(w,cn);
                    
                    %ignore zeros for speed
                    bad = wr<=0;
                    wr = wr(~bad);
                    yr = y(~bad,:);

                    % density
                    [dr,~,~] = ksdens(yr,pts,bw_embed,'weights',wr);
                    dens_rate_rand(:,:,ir) = dr;
                    %kr(ir) = kldiv_nd(dr,dens_embed);
                end
            else
                dens_rate_rand = [];
            end
            
            % store
            tmpres = [];
            tmpres.name = name;
            tmpres.dens_rate = dens_rate ./ nansum(dens_rate(:));
            tmpres.dens_rate_rand = dens_rate_rand ./ sum(sum(dens_rate_rand,1),2);
            tmpres.dens_embed = dens_embed ./ nansum(dens_embed(:));
            tmpres.nsmp = [size(y,1) size(yr,1)];

            % debug
            if 0
                
            figure
            nr = 2; nc = 2;
            subplot(nr,nc,1); imagesc(dens_rate); axis square; colorbar; title('rate')
            subplot(nr,nc,2); imagesc(dens_embed); axis square; colorbar; title('embed')
            subplot(nr,nc,3); imagesc(dens_rate-dens_embed); axis square; colorbar; title('rate-embed')
            
            end
            
            % cosine similarity
            if getSampleSimilarity
                fprintf('map %g, similarity\n',ii)
                
                % prep   
                idx=1:2:numel(c);
                
                xx = [y, sdf'];
                xx = double(xx);
                xx_orig = zscore(xx);  
                xx_orig=xx_orig(idx,:);
                xx_orig = gpuArray(xx_orig);
                c2 = c(idx);
                
                nsmp = accumarray(c2,1,[nstate 1]);
                
                Dn = nan(nstate,nstate);
                D = nan(nstate,nstate);
                D_rand = nan(nstate,nstate,nrand);

                % calculate for each combo of poses
                tmp = 1:nstate;
                tmp(nsmp<5)=[];
                cmb = combnk(tmp,2);
                
                for ic=1:size(cmb,1)
                    %tic
                    is1 = cmb(ic,1);
                    is2 = cmb(ic,2);
                    
                    % lower sample makes everything faster
                    if nsmp(is2) < nsmp(is1)
                        is1 = cmb(ic,2);
                        is2 = cmb(ic,1);
                    end
                    
                    % sort for easy within-cluster index determination
                    xx = [xx_orig(c2==is1,:); xx_orig(c2==is2,:)];
                    d = pdist(xx);
                    
                    % compute within-cluster indices
                    n = size(xx,1);
                    ndist = nchoosek(n,2);
                    nw1 = nchoosek(nsmp(is1),2);
                    nw2 = nchoosek(nsmp(is2),2);
                    nb = ndist-nw1-nw2;
                    
                    iwith1 = false(1,ndist);
                    s=-n+1;
                    for is=1:nsmp(is1)-1
                        s = s+n-is+1;
                        f = s+(nsmp(is1)-is)-1;
                        iwith1(s:f) = 1;
                    end
                    
                    s = nw1+nb+1;
                    iwith2 = false(1,ndist);
                    iwith2(s:end) = 1;
                    
                    % precompute (careful for overflow)
                    st = sum(d);
                    sw1 = sum(d.*iwith1);
                    sw2 = sum(d.*iwith2);
                    sb = st-sw1-sw2;
                    d=[]; % clean
                    
                    % store
                    Dn(is1,is1) = nw1;
                    D(is1,is1) = gather(sw1 ./ nw1);
                    Dn(is2,is2) = nw2;
                    D(is2,is2) = gather(sw2 ./ nw2);
                    Dn(is1,is2) = nb;
                    D(is1,is2) = gather(sb./nb);
                
                    % now randomize
                    if nrand > 0
                        for ir=1:nrand
                            idx = randperm(n);
                            xr = [xx(:,1:2), xx(idx,3)];
                            dr = pdist(xr);

                            % precompute
                            st = sum(dr);
                            sw1 = sum(dr.*iwith1);
                            sw2 = sum(dr.*iwith2);
                            sb = st-sw1-sw2;
                            dr=[]; % clean

                            % store
                            D_rand(is1,is1,ir) = gather(sw1 ./ nw1);
                            D_rand(is2,is2,ir) = gather(sw2 ./ nw2);
                            D_rand(is1,is2,ir) = gather(sb ./ nb);
                        end
                    end
                    %toc
                end
                
                % store
                tmpres.D = D;
                tmpres.Dn = Dn;
                tmpres.D_rand = D_rand;
            end
            toc
            
            % save
            parsave(sname,tmpres)
            foo=1;
        end
        

        % clean
        disp('cleaning...')
        delete(gcp('nocreate'))
        delete SDF_par Y_par frame_par C_par
        
        toc(startTime)
    end
    
    % load back the maps
    fprintf(' loading back maps')

    res_emb_rate = [];
    for ii=1:numel(SDF)
        dotdotdot(ii,0.1,numel(SDF))

        name = sdfnames{ii};
        sname = [mapdir '/' name(1:end-12) '_maps.mat'];
        tmp = load(sname);
        res_emb_rate = cat(1,res_emb_rate,tmp);
    end
end



%% summary
if plotResidResults
    if loadFitResults
        res_mdl = [];
        for ii=1:numel(SDF)
            dotdotdot(ii,0.1,numel(SDF))
            name = [SDF(ii).name(1:end-16) '_regmdl.mat'];
            
            in = load([sdfpath_data '/' name]);
            
            res_mdl = cat(1,res_mdl,in);
        end
    end
    
    % extrat
    D = cellfun(@(x) x.Deviance,{res_mdl.Rsquared});
    D2 = [res_mdl.Deviance];
    R = [SDF.mean_rate];
    
    
    % ------------------------------------------------
    % plot
    r = [];
    p = [];
    str = sprintf('rate vs deviance, Spearman\n');
    for ia=1:numel(uarea)
        sel=iarea==ia;
        [a,b] = corr(R(sel)',D(sel)','type','spearman');
        if mod(ia,3)==0
            str=sprintf('%s\n %s: R=%.3g, p=%.3g',str,uarea{ia},r,p);
        else
            str=sprintf('%s ||| %s: R=%.3g, p=%.3g',str,uarea{ia},r,p);
        end
        r(ia) = a;
        p(ia) = b;
    end
    
    % plot all
    fig = figure;
    set_bigfig(gcf,[0.5 0.8])
    
    hs = scatterhist(R,D,'marker','.','markersize',10,'parent',fig,'NBins',100,'Group',areas,'plotgroup',1,'kernel','on');
    title(str)
    xlabel('mean rate')
    ylabel('deviance')
    set(hs(1),'xscale','log')
    set(hs(2),'xscale','log')
    
    % save
    sname = [figdir '/resid_summary.pdf'];
    save2pdf(sname,gcf);
    
    foo=1;
end


%% RSA between transitions and firing rate diffs
if plotTransitionRSA
    % settings
    lags = [1:2:20];
    
    nrand_rsa = 25;
    corrtype = 'spearman';
    
    % path
    tmp = {'noresid','resid'};
    rsadir = [sdfpath '/rsa_' tmp{useSDF_resid+1}];
    if ~exist(rsadir); mkdir(rsadir); end
    
    % recalc?
    if getTransitionRSA
        
        % split for parallel processing
        Y_par = {};
        frame_par = {};
        C_par = {};
        for id=1:numel(datasets)
            seldat = idat==id;
            frame_par{id} = frame(seldat);
            Y_par{id} = Y_test(seldat,:);
            C_par{id} = C(seldat);
        end
        
        % loop
        res_rsa = nan(numel(SDF),1);
        res_rsa_rand = nan(numel(SDF),numel(lags),nrand_rsa);
        tic
        fprintf('getting  trans prob rsa')
        %parfor ii=1:numel(SDF)            
        for ii=1:numel(SDF)    
            dotdotdot(ii,0.1,numel(SDF))
            %fprintf('rsa %g \n',ii)
            name = sdfnames{ii};
            datname = name(1:23);
            sdf_in = SDF(ii);
            id = sdf_in.id;

            sname = [rsadir '/' name(1:end-12) '_rsa.mat'];
            
            % spike info
            f1 = sdf_in.frame;
            sdf = sdf_in.sdf;

            % pose info
            [~,ia,ib] = intersect(frame_par{id},f1);
            
            f2 = frame_par{id}(ia);
            y = Y_par{id}(ia,:);
            c = C_par{id}(ia);
            sdf = sdf(ib);
            
            % collapse poses
            [rate,c_state,istate] = collapse_poses(sdf,c,winlim);
            [y,~,istate] = collapse_poses(y,[],winlim,istate);
                        
            for ilag=1:numel(lags)
                lag = lags(ilag);
                
                % get transitions and RSA
                st = 1:1:numel(rate)-lag;
                fn = lag+1:1:numel(rate);
                dRate = rate(fn) - rate(st);
                
                if rsaUseDist
                    if lag~=1; error('fix'); end
                    dist = (y(2:end,:) - y(1:end-1,:)).^2;
                    dist = sqrt(sum(dist,2));
                    w = dist;
                else
                    w = 1;
                end
                po = cond_trans_prob(c_state,lag,nstate,w);
                por = cond_trans_prob(c_state,lag,nstate,dRate);
                %bad = po<0.01;
                bad = false(size(po));
                p1 = po(~bad);
                p2 = por(~bad);
                r = corr([p1(:)],[p2(:)],'type',corrtype);

                % now randomize
                rr = nan(1,nrand_rsa);
                for ir=1:nrand_rsa
                    %idx = randperm(numel(rate));
                    idx = ceil([0.25 0.75]*numel(rate));
                    idx = circshift(1:numel(rate),idx);
                    rater = rate(idx);
                    por = cond_trans_prob(c_state,1,nstate,diff(rater));

                    p1 = po(~bad);
                    p2 = por(~bad);
                    rr(ir) = corr([p1(:)],[p2(:)],'type',corrtype);            
                end

                % store
                res_rsa(ii,ilag) = r;
                res_rsa_rand(ii,ilag,:) = rr;
            end
        end
        toc
        
        res_rsa_orig = res_rsa;
        res_rsa_rand_orig = res_rsa_rand;
        % save
    end
    res_rsa = res_rsa_orig;
    res_rsa_rand = res_rsa_rand_orig;
    
    % ------------------------------

    % plot for lag==1
    if 1
        rsa = res_rsa;
        rsar = nanmean(res_rsa_rand,2);
    else
        rsa = abs(res_rsa);
        rsar = nanmean(abs(res_rsa_rand),2);
    end
    
    figure
    nr = 1; nc = 1; ns=0;
    
    [mur,ser] = avganderror_group(iarea,rsar,'mean',200);
    
    ns = ns+1;
    subplot(nr,nc,ns)
    cfg = [];
    cfg.dobar = 0;
    cfg.groups = uarea;
    cfg.ylabel = 'mean corr';
    cfg.title = 'RSA: transition prob and rate diff';
    out1 = plot_anova(rsa,iarea,cfg);  
    hold all
    hb = errorbar(out1.xx,mur',ser','k.-','markersize',10);
    
    legend([out1.hbar hb],{'obs','rand'})
    
end

%% mean rate per pose
if plotMeanRate
    avgtype = 'mean';
    
    % prep stuff
    rate = cat(3,RES_rate(:).rate_pose);
    rate_sd = cat(3,RES_rate(:).rate_pose_sd);
    rate = permute(rate(:,1,:),[3 1 2]);
    rate_sd = permute(rate_sd(:,1,:),[3 1 2]);
    rate = (rate-nanmean(rate)) ./ nanstd(rate);
    
    xx = 1:nstate;
    
    % plot
    figure
    nr = 2; nc = 4;ns=0;
    set_bigfig(gcf,[0.35 0.3])
    cols = get_safe_colors(0);

    % plot mean rates
    [mu,se] = avganderror(rate,avgtype,1,1,200);

    ns=ns+1;
    subplot(nr,nc,ns)
    errorbar(xx,mu,se,'k.-')
    
    title('rates per pose')
    xlabel('pose')
    ylabel([avgtype ' norm rate'])
    axis square
    
    % plot mean rates per area
    [mu,se] = avganderror_group(iarea,rate,avgtype,200);

    ns=ns+1;
    subplot(nr,nc,ns)
    h = [];
    for ia=1:numel(uarea)
        h(ia)=errorbar(xx,mu(ia,:),se(ia,:),'.-','color',cols(ia,:));
        hold all
    end
    
    title('mean rates per pose')
    xlabel('pose')
    ylabel([avgtype ' norm rate'])
    axis square
    hl = legend(h,uarea);
    pos1 = get(hl,'position');
    pos2 = [0.005 pos1(2:4)];
    set(hl,'position',pos2)
    
    % plot overall rates
    tmp = nanmean(rate,2);
    
    ns=ns+1;
    subplot(nr,nc,ns)
    cfg = [];
    cfg.dobar = 0;
    cfg.groups = uarea;
    cfg.ylabel = [avgtype ' overall rate'];
    cfg.title = 'mean rate';
    out1 = plot_anova(tmp,iarea,cfg); 
    
    % plot rate SD
    tmp = nanstd(rate,[],2);
    
    ns=ns+1;
    subplot(nr,nc,ns)
    cfg = [];
    cfg.dobar = 0;
    cfg.groups = uarea;
    cfg.ylabel = [avgtype ' mean rate SD'];
    cfg.title = 'neuronal pose tuning';
    out1 = plot_anova(tmp,iarea,cfg); 
    
    % plot mean SD
    [mu,se] = avganderror(rate_sd,avgtype,1,1,200);

    ns=ns+1;
    subplot(nr,nc,ns)
    errorbar(xx,mu,se,'k.-')
    
    title('rate SD per pose')
    xlabel('pose')
    ylabel([avgtype ' SD'])
    axis square
    
    % plot mean SD per area
    [mu,se] = avganderror_group(iarea,rate_sd,avgtype,200);

    ns=ns+1;
    subplot(nr,nc,ns)
    h = [];
    for ia=1:numel(uarea)
        h(ia)=errorbar(xx,mu(ia,:),se(ia,:),'.-','color',cols(ia,:));
        hold all
    end
    
    title('SD per pose')
    xlabel('pose')
    ylabel('mean SD')
    axis square

    % plot overall SD
    tmp = nanmean(rate_sd,2);
    
    ns=ns+1;
    subplot(nr,nc,ns)
    cfg = [];
    cfg.dobar = 0;
    cfg.groups = uarea;
    cfg.ylabel = [avgtype ' overall rate SD'];
    cfg.title = 'neuronal pose tuning';
    out1 = plot_anova(tmp,iarea,cfg); 
    
    % save
    sname = [figdir '/mean_rates.pdf'];
    save2pdf(sname,gcf);
end


%% plot tuning, as a function of engagment
if plotMeanRate_engage
    avgtype = 'mean';
    
    rate_sd = cat(3,RES_rate(:).rate_pose_sd);
    rate_sd = permute(rate_sd,[3 1 2]);
    rate_sd = rate_sd(:,:,2:end);
    rate_sd = squeeze( nanmean(rate_sd,2) );
    
    [mu,se] = avganderror_group(iarea,rate_sd,avgtype);
    xx = 1:numel(uarea);
    
    figure
    hp = errorbar(repmat(xx',1,size(mu,2)),mu,se);
    hl=legend(hp,{'no','yesLow','yesHigh'},'location','eastoutside');
    hl.Title.String = 'engagment';
    
end

%% plot classification resuluts
if plotClassification_pose
    dc = dir([clfpath '/*clf.mat']);
    
    % load and concatenate
    if loadPoseClassification

        res_clf= [];
        for id=1:numel(dc)
            fprintf('%g, ',id)
            name = dc(id).name;
            in = load([clfpath '/' name]);

            % reformat coef stuff
            if iscell(in.coef_obs)
                mu = cellfun(@(x) mean(x), in.coef_obs,'un',0);
                mu = cat(1,mu{:}); % fold x feat
            else
                if strcmp(classifierType,'svc')
                    mu = nanmean(in.coef_obs,2);
                    mu = permute(mu,[1 3 2]);
                elseif strcmp(classifierType,'xgb')
                    mu = in.coef_obs;
                end
                    
                foo=1;
            end
            in.coef_obs = mu;
            
            if iscell(in.coef_rand)
                mur = cellfun(@(x) mean(x), in.coef_rand,'un',0);
                mur = reshape( cat(1,mur{:}), [size(in.coef_rand), size(mu,2)] );
                mur = permute(mur,[2 3 1]);
            else
                if strcmp(classifierType,'svc')
                    mur = nanmean(in.coef_rand,3);
                    mur = permute(mur,[2 4 1 3]);
                elseif strcmp(classifierType,'xgb')
                    mur = permute(in.coef_rand,[2 3 1]);
                end
                foo=1;
            end
            in.coef_rand = mur;
            
            % append
            res_clf = cat(1,res_clf,in);

            foo=1;
        end
        fprintf('\n')
    end
    
    % compile results
    N = cellfun(@numel,{res_clf.area})';
    
    if 1
        A = cat(1,res_clf(:).accuracy_obs);
        Ar = permute( cat(3,res_clf(:).accuracy_rand), [3 2 1] );
        astr = 'balanced-accuracy';
    else
        A = cat(1,res_clf(:).F1_obs);
        Ar = permute( cat(3,res_clf(:).F1_rand), [3 2 1] );
        astr = 'balanced-F1';
    end
    
    % coefficients per area
    carea = [res_clf.area];
    carea = collapse_areas(carea);
    [G,~,iG] = unique(carea);
    
    coef = cat(2,res_clf(:).coef_obs);
    coefr = cat(2,res_clf(:).coef_rand);
    
    coef = abs(coef);
    coefr = abs(coefr);
    
    % ------------------------------------------------
    % start figure
    figure
    nr = 1; nc = 3;
    set_bigfig(gcf,[0.6 0.25])
    set(gcf,'units','normalized')
    pos = get(gcf,'position');
    set(gcf,'position',[0.4 0.27 pos(3:4)])
        
    % plot accuracy
    mu = nanmean(A,2);
    mur = mean(mean(Ar,3),2);
    a = mu - mur;
    [mua,sea] = avganderror(a,'mean');
    
    subplot(nr,nc,1)
    h1 = histogram(mu,10,'normalization','probability');
    hold all
    h2 = histogram(mur,10,'normalization','probability');
    
    axis square
    legend({'obs','rand'},'location','northeast')

    str = sprintf('excess %s\n%.3g+ %.3g',astr,mua,sea);
    title(str)
    xlabel(astr)
    ylabel('prop')
    set(gca,'fontsize',fontSize)
    
    % plot influence per area
    mu = nanmean(coef);
    mur = mean(mean(coefr,3),1);
    
    if 1
        R = mu - mur;
    else
        R = mu;
    end
    mu = grpstats(R, iG, {@mean});
    se = grpstats(R, iG,{@(x) nanstd(bootstrp(200,@(x) mean(x),x))});
    [p, atab] = anovan(R',iG,'display','off');
    
    xx = 1:numel(G);
    
    subplot(nr,nc,2)
    barwitherr(se,xx,mu)
    
    str = sprintf('excess influence (obs-rand coef) per area\nanova F=%.3g, p=%.3g',atab{2,6},p);
    title(str)
    ylabel('excess influence (obs-rand coef)')
    
    axis square
    set(gca,'xtick',xx,'xticklabel',G);
    set(gca,'fontsize',fontSize)

    % correlate accuracy with number of cells
    mu = nanmean(A,2);
    mur = mean(mean(Ar,3),2);
    R = mu - mur;
    
    [r,p] = corr(R,N);
    subplot(nr,nc,3)
    scatter(R,N)
    
    axis square
    lsline
    
    str = sprintf('corr excess %s vs number of cells\nR=%.3g, p=%.3g',astr,r,p);
    title(str)
    xlabel(['excess ' astr])
    ylabel('# cells')
    
    foo=1;
    set(gca,'fontsize',fontSize)
    
    % save
    sname = [figdir '/classify_pose_' astr '.pdf'];
    save2pdf(sname,gcf)
    
    % ---------------------------------------------
    % plot correlation, rate vs accuracy
    %figure('name','rate vs acc')
    
    %sname = [figdir '/classify_pose.pdf'];
    %save2pdf(sname,gcf)
end


%% classification by module

%% plot classification resuluts
if plotClassifyMod
    thisMet = 'F1';
    avgtype = 'mean';
    
    dc = dir([clfmodpath '/*clf.mat']);
    
    % load and concatenate
    if loadClassifyMod

        res_modclf = [];
        for id=1:numel(dc)
            fprintf('%g, ',id)
            name = dc(id).name;
            in = load([clfmodpath '/' name]);

            % reformat coef stuff
            for ii=1:numel(in.out)
                tmp = in.out(ii);

                if iscell(tmp.coef_obs)
                    mu = cellfun(@(x) mean(x,1), tmp.coef_obs,'un',0);
                    mu = cat(1,mu{:}); % fold x feat
                else
                    if strcmp(classifierType,'svc')
                        mu = nanmean(tmp.coef_obs,2);
                        mu = permute(mu,[1 3 2]);
                    elseif strcmp(classifierType,'xgb')
                        mu = tmp.coef_obs;
                    end

                    foo=1;
                end
                tmp.coef_obs = mu;
            
                if iscell(tmp.coef_rand)
                    mur = cellfun(@(x) mean(x,1), tmp.coef_rand,'un',0);
                    mur = reshape( cat(1,mur{:}), [size(tmp.coef_rand), size(mu,2)] );
                    mur = permute(mur,[2 3 1]);
                else
                    if strcmp(classifierType,'svc')
                        mur = nanmean(tmp.coef_rand,3);
                        mur = permute(mur,[2 4 1 3]);
                    elseif strcmp(classifierType,'xgb')
                        mur = permute(tmp.coef_rand,[2 3 1]);
                    end
                    foo=1;
                end
                tmp.coef_rand = nanmean(mur,3);
                
                in.out(ii) = tmp;
            end
            
            
            % append            
            res_modclf = cat(1,res_modclf,in);

            foo=1;
        end
        fprintf('\n')
    end
    
    % cull?
    na = cellfun(@numel,{res_clf.area})';
    
    if 0
        nthresh = 30;
        del = na < nthresh;
        
        res_modclf(del) = [];
        na(del) = [];
    end
    
    nmx = max( cellfun(@numel,{res_modclf.out}) );
    ndat = numel(res_modclf);

    % compile results
    A = []; % dataset
    Ar = [];
    coef = nan(sum(na),nmx);
    coefr = nan(sum(na),nmx);
    
    fn = 0;
    for ii=1:numel(res_modclf)
        st = fn+1;
        fn = fn + numel(res_modclf(ii).area);
        tmp = res_modclf(ii).out;
        for jj=1:numel(tmp)
            if strcmp(thisMet,'F1')
                A(ii,jj) = nanmean(tmp(jj).F1_obs);
                Ar(ii,jj) = nanmean(tmp(jj).F1_rand(:));
                astr = 'F1';
            else
                A(ii,jj) = nanmean(tmp(jj).accuracy_obs);
                Ar(ii,jj) = nanmean(tmp(jj).accuracy_rand(:));
                astr = 'balanced acc';
            end
            
            if 1 % mean influence
                c = nanmean(nanmean(tmp(jj).coef_obs,3),1);
                cr = nanmean(nanmean(tmp(jj).coef_rand,3),1);

                c = abs(c);
                cr = abs(cr);
                
                if 1
                    c = c ./ max(c);
                    cr = cr ./ max(cr);
                end
                
                foo=1;
            else % ranked influence
                c = tmp(jj).coef_obs;
                cr = tmp(jj).coef_rand;
                
                c = abs(c);
                cr = abs(cr);
                
                [~,c] = sort(c,2,'descend');
                [~,cr] = sort(cr,2,'descend');
                
                c = nanmean( c ./ size(c,2), 1 );
                cr = nanmean( cr ./ size(cr,2), 1 );
                
                foo=1;
            end
            

            
            % store
            coef(st:fn,jj) = c;
            coefr(st:fn,jj) = cr;
        end
    end
    
    
    if 1
        A = A - Ar;
        Ar = nan(size(A));
    end
    
    % area
    carea = [res_modclf.area];
    carea = collapse_areas(carea);
    [G,~,iG] = unique(carea);
    
    % ------------------------------------------------
    % plot
    xx = res_modclf(1).Cutoffs;
    %xx(end-1) = [];
    
    figure;
    nr = 1; nc = 2;
    set_bigfig(gcf,[0.4 0.25])
    set(gcf,'units','normalized')
    pos = get(gcf,'position');
    set(gcf,'position',[0.4 0.27 pos(3:4)])
    
    % plot accuracy
    [mu,se] = avganderror(A,avgtype);
    [mur,ser] = avganderror(Ar,avgtype);

    subplot(nr,nc,1)
    h = [];
    htmp = shadedErrorBar(xx,mu,se,{'r-'});
    h(1) = htmp.mainLine;
    hold all
    htmp = shadedErrorBar(xx,mur,ser,{'k-'});
    h(2) = htmp.mainLine;

    title('decoding performance per session')
    xlabel('nclust')
    ylabel([avgtype ' excess ' astr])
    legend(h,{'obs','rand'})
    axis square
    
    % coefficients
    %tmpc = coef - coefr;
    tmpc = coef;
    [mu,se] = avganderror_group(iG,tmpc,avgtype);
    %[mur,ser] = avganderror_group(iG,coefr,avgtype);
    
    subplot(nr,nc,2)
    hp = plot(xx,mu','.-');
    %hold all
    %plot(xx,mur','--');

    title('influence of each area on decoding')
    xlabel('nclust')
    ylabel([avgtype ' coef diff (=obs-rand)'])
    legend(G)
    axis square
    
end


%% plot accuracy as a function of rate and rate SD
if plotCorrAccRate
    corrtype = 'spearman';
    
    % rate stuff
    rate = cat(3,RES_rate(:).rate_pose);
    rate_sd = cat(3,RES_rate(:).rate_pose_sd);
    rate = permute(rate(:,1,:),[3 1 2]);
    rate_sd = permute(rate_sd(:,1,:),[3 1 2]);
    rate = (rate-nanmean(rate)) ./ nanstd(rate);
       
    % compile for different datasets
    tmp = cellfun(@(x) x(1:end-9),{datasets.name},'un',0);
    idx = nan(size(sdfnames));
    for id=1:numel(datasets)
        name = tmp{id};
        seld = contains(sdfnames,name);
        idx(seld) = id;
    end
     
    % accuracy stuff
    A = cat(1,res_clf(:).accuracy_obs);
    Ar = permute( cat(3,res_clf(:).accuracy_rand), [3 2 1] );
    A = nanmean(A,2) - nanmean(nanmean(Ar,3),2);
    A = A(idx,:);
    
    % fit the model
    if 0
    tmpr = nanstd(rate,[],2);
    tmps = nanstd(rate_sd,[],2);
    X = [tmpr,tmps];
    X = (X-nanmean(X)) ./ nanstd(X);
    mdl = fitlm(X,A);
    
    % plot results
    b = mdl.Coefficients.Estimate;
    bp = mdl.Coefficients.pValue;
    [p,f] = coefTest(mdl);
    
    figure
    bar(1:2,b(2:3))
    
    str = sprintf('model F=%.3g, p=%.3g\nrates: b=%.3g, p=%.3g\n std: b=%.3g, p=%.3g',...
        f,p,b(2),bp(2),b(3),bp(3));
    title(str)
    ylabel('beta')
    set(gca,'xtick',1:2,'xticklabel',{'rate devs','std devs'});
    
    sname = [figdir '/mdl_acc_rateStdDevs.pdf'];
    save2pdf(sname,gcf)
    end
    
    tmpr = nanstd(rate,[],2);
    tmps = nanstd(rate_sd,[],2);
    %tmpr = mad(rate,0,2);
    %tmps = mad(rate_sd,0,2);
    
    RR = [];
    RS = [];
    P = [];
    for ia=1:numel(uarea)
        sel = find(iarea==ia);
        [rr,pr] = corr(tmpr(sel),A(sel),'type',corrtype);
        [rs,ps] = corr(tmps(sel),A(sel),'type',corrtype);
        
        if 0
            n = numel(sel);
            z1 = atanh(rr);
            z2 = atanh(rs);
            zobs = (z1-z2) / sqrt( 1 / (n-3) + 1 / (n-3) );
            pval = 2 * normcdf(zobs);
        else
            % significance
            do = rs - rr;

            dr = [];
            for ib=1:100
                idx = randperm(numel(sel));
                idx = sel(idx);
                a1 = tmpr(idx);
                a2 = tmps(idx);

                [r1,pr] = corr(a1,A(sel),'type',corrtype);
                [r2,ps] = corr(a2,A(sel),'type',corrtype);

                dr(ib) = r2 - r1;
            end
            pval = sum( abs(dr) > abs(do)) ./ numel(dr);

        end
    
    
        % sttore
        RR(ia,1) = rr;
        RS(ia,1) = rs;
        P(ia) = pval;
    end
    
    mup = ones(size(RR)) * max([RR;RS])*1.05;
    mup(P>0.05) = nan;
    
    % plot
    figure
    bar([RR,RS]);
    hold all
    plot(1:numel(uarea),mup,'k.','markersize',10)
    
    title('correlation of activty with session accuracy')
    ylabel([corrtype ' R'])

    axis square
    set(gca,'xtick',1:numel(uarea),'xticklabel',uarea)
    legend({'rate diff','std diff'},'location','eastoutside')
    
    % save
    sname = [figdir '/corr_acc_rateStdDevs.pdf'];
    save2pdf(sname,gcf)
end

%% peri-transition PSTHs
if plotPSTH_pose
    getSegments = 0;
        onlyNonEngage = 1;
        
    lim = [-1 1] * 1; %sec
    minSeg = 0.2;
    smoothWin = ceil(1*fs_frame);
    
    normtype = 'presegnorm';
        doWeightedMean = 1;
    avgtype = 'median';
    
    if getSegments
        RES_seg = [];

        % get all segments
        for id=1:numel(datasets)
            name = datasets(id).name(1:end-9);
            fprintf('%g: %s, getting segment...\n',id,name);

            % get SDF
            day = datestr( name(4:end-10) );
            selsdf = ismember({SDF.day},day);
            sdf = SDF(selsdf);

            % concate rates
            Xs = cat(1,sdf.sdf);
            
            if 0
                Xs = zscore(Xs,[],2);
            end

            % get state stuff
            sel = idat==id;
            c = C(sel);
            f = frame(sel);

            % state indices
            istate = find( diff(c)~=0 );
            
            % get rid of too short segs
            d = [diff(istate); numel(c) - istate(end)+1];
            tooShort = d < minSeg*fs_frame;
            istate(tooShort) = [];
            
            % create indices
            s1 = abs(lim(1)*fs_frame);
            s2 = abs(lim(2)*fs_frame);
            istate(istate <= s1) = [];
            istate(istate + s2 > numel(c)) = [];
            
            idx = -s1:s2;
            idx = [ repmat(idx,numel(istate),1) + istate ]';
            idx = [idx(:)];
                        
            % extract
            ncell = sum( cellfun(@numel,{sdf.area}) );
            ndat = numel(istate);
            nsmp = diff(lim*fs_frame)+1;
            
            xs = Xs(:,idx);
            xs = reshape(xs,[ncell,nsmp,ndat]);
            S = permute(xs,[3 1 2]);
            
            C_seg = [c(istate), c(istate+1)];
            
            %only non-engaged periods?
            if onlyNonEngage
                % task engagement
                datfold = [spkparentpath  '/' name];
                [eng,f_eng] = get_task_engagement(datfold,smoothWin,f);

                % clean
                idx2 = reshape(idx,nsmp,ndat);
                badidx = false(size(idx2));
                badidx(f(idx2) < f_eng(1)) = 1;
                badidx(f(idx2) > f_eng(end)) = 2;
                
                tmpidx = find(ismember(f(idx),f_eng),1);
                tmpidx = idx(tmpidx);
                idx2(badidx==1) = tmpidx;
                idx2(badidx==2) = tmpidx;
                
                % find engaged segments
                tmp = eng(idx2);
                bad = any( tmp > 0 | badidx~=0 );
                
                % cull
                istate(bad) = [];
                S(bad,:,:) = [];
                C_seg(bad,:) = [];
            end
            
            % save and store
            tmp = [];
            tmp.name = name;
            tmp.C_seg = C_seg;
            tmp.seg = S;
            tmp.area = collapse_areas([sdf.area]);

            if 0
                sname = [tmpdir '/' name '_seg.mat'];
                save(sname,'-struct','tmp');
            else
                RES_seg = cat(1,RES_seg,tmp);
            end
            
            foo=1;
        end
        
    end
    
    % prep
    %A = [RES_seg.area];
    %[ua,~,ia] = unique(A);
    
    st = lim(1)*fs_frame;
    fn = lim(2)*fs_frame;
    t_lim = [st:fn] ./ fs_frame;
    
    foo=1;
    % get mean rate per area
    
    MU = [];
    MU2 = [];
    for id=1:numel(RES_seg)
        tmp = RES_seg(id).seg;
        pre = tmp(:,:,t_lim<0);
        post = tmp(:,:,t_lim>0);
        c = RES_seg(id).C_seg(:,2); % pre OR post state
        
        % time series
        if strcmp(normtype,'presegnorm') % z-norm by pre of each seg
            m = nanmedian(pre,3);
            s = mad(pre,1,3);

            if doWeightedMean % weighted mean
                [uc,~,ic] = unique(c);
                tmpm = [];
                for ii=1:numel(uc)
                    sel = ic==ii;
                    a = nanmedian( (tmp(sel,:,:) - m(sel,:)) ./ s(sel,:),1 );
                    tmpm = cat(1,tmpm,a);
                end
                mu = nanmedian(tmpm,1);
                foo=1;
            else
                mu = nanmedian( (tmp-m)./s );        
                %mu = nanmean( (tmp-m)./s );  
            end
        elseif 0 % norm by pre of each seg
            m = nanmean(pre,3);
            mu = tmp - m;
            mu = nanmean(mu,1);
        elseif 0 % z-norm by mean of all pre-activity
            %mu = nanmean(tmp);
            m = nanmean(pre(:));
            s = nanstd(pre(:));
            mu = (tmp-m) ./ s;
            mu = nanmean(mu);
        elseif 0 % z-norm by all activity
            tmp2 = tmp;
            for ii=1:size(tmp,2)
                tmp2(:,ii,:) = zscore_robust(tmp(:,ii,:),[],'all');
            end
            mu = nanmedian(tmp2);
        elseif 0 % z-norm each seg
            mu = nanmedian( zscore(tmp,[],3) );
            
        elseif 0 % norm to [0 1]
            %mu = tmp ./ sum(abs(tmp),3);
            mu = tmp + min(tmp,[],3);
            mu = mu ./ sum(tmp,3);
            mu = nanmean(mu);            
        else
            mu = nanmean(tmp);
        end
        mu = squeeze(mu);

        if any(abs(mu(:))==Inf)
            foo=1;
        end
        MU = cat(1,MU,mu);
        
        % pure pre/post
        p1 = nanmean(pre,3);
        p2 = nanmean(post,3);
        a = (p2-p1) ./ (p2+p1);
        a = nanmean(a,1)';
        MU2 = cat(1,MU2,a);
    end
    
    % clean
    bad = isnan(MU) | abs(MU)>10^10;
    MU(bad) = nan;
    
    % norm by area here?
    if 0
        MUtmp = MU;
        for ia=1:numel(uarea)
            sela = iarea==ia;
            bad = any(isnan(MU),2) | any(abs(MU)==Inf,2);
            
            sel = sela & ~bad;
            pre = MU(sel,t_lim<0);
            
            mu = nanmean(pre(:));
            se = nanstd(pre(:));

            tmp = MU(sela,:);
            tmp = (tmp-mu) ./ se;
            MU(sela,:) = tmp;
        end
    end
    
    % start figure
    figure;
    nr = 1; nc = 2;
    set_bigfig(gcf,[0.3 0.2])
    cols = get_safe_colors(0,[1:5 7]);
    
    
    % plot mean timesries, split by area
    [mu,se] = avganderror_group(iarea,MU,avgtype,100);
    
    p = [];
    for ii=1:size(MU,2)
        if strcmp(avgtype,'median')
            p(ii) = kruskalwallis(MU(:,ii),iarea,'off');
        else
            p(ii) = anovan(MU(:,ii),iarea,'display','off');
        end
    end
    %p = p * numel(p);
    p = bonf_holm(p);
    
    mup = ones(size(t_lim)) * max(mu(:)) * 1.05;
    mup(p>0.05) = nan;
    
    subplot(nr,nc,1)
    h = [];
    for ii=1:size(mu,1)
        if 1
            htmp = shadedErrorBar(t_lim,mu(ii,:),se(ii,:),{'-','color',cols(ii,:)},0);
            h(ii) = htmp.mainLine;
        else
            h(ii) = plot(t_lim,mu(ii,:),'color',cols(ii,:));
        end
        hold all
    end
    plot(t_lim,mup,'k.')
    
    pcl('x',0)
    legend(h,uarea,'location','northwest')
    
    title('mean rate, normalized to pre switch')
    xlabel('time')
    ylabel([avgtype ' baseline-norm rate'])
    
    axis square
    
    % plot pre/post
    if 1
        sel0 = find(t_lim==0);
        pre = nanmean(MU(:,t_lim < 0),2);
        post = nanmean(MU(:,t_lim > 0),2);
        d = post - pre;
        [mu,se] = avganderror_group(iarea,d,avgtype,100);
        [~,T] = kruskalwallis(d,iarea,'off');
        fa = T{2,5};
        pa = T{2,6};
        
        p = [];
        for ia=1:numel(uarea)
            sela = iarea==ia;
            tmp = d(sela);
            p(ia) = signrank(tmp);
        end
        mup = ones(size(mu)) * max(mu+se)*1.05;
        mup(p>0.05) = nan;
        
        subplot(nr,nc,2)
        hb=[];
        for ia=1:numel(uarea)
            hb(ia) = barwitherr(se(ia),ia,mu(ia));
            set(hb(ia),'facecolor',cols(ia,:));
            hold all
        end
        hold all
        plot(1:numel(uarea),mup,'k.')

        s = sprintf('post-pre diff\nKW X2=%.3g, p=%.3g',fa,pa);
        title(s)
        ylabel([avgtype ' norm post-pre diff'])

        set(gca,'xtick',1:numel(uarea),'xticklabel',uarea)
        axis square
    end
    
    
    % save
    strs1 = {'all','nonengage'};
    sname = sprintf('%s/psth_switch_%s_%s_weighted%g.pdf',figdir,strs1{onlyNonEngage+1},normtype,doWeightedMean);
    save2pdf(sname)
    
    foo=1;

    %}
end

%% correlate rate and pose residence
if plotCorrRateResidence
    % stuff
    corrtype = 'spearman';
    
    R = cat(2,RES_rate(:).rate_pose)';
    B = cat(2,RES_rate.pose_residence)';
    
    % normalize
    mu = nanmean(R,2);
    se = nanstd(R,[],2);
    R = (R-mu)./se;

    B = B ./ nansum(B,2) * 100;

    % correlate
    a = repmat(iarea,1,size(B,2));
    x = [B(:)];
    y = [R(:)];
    a = [a(:)];
    bad = isnan(x) | isnan(y);
    x(bad) = [];
    y(bad) = [];
    a(bad) = [];
    [r,p] = corr(x,y,'type',corrtype);
    [rg,pg]=corr_group(iarea,x,y,'type',corrtype);
    
    % plot
    figure
    h = scatterhist(x,y,'group',a,'kernel','on');
    
    str = sprintf('pose residence vs rate\n%s R=%.3g, p=%.3g',corrtype,r,p);
    title(str)
    xlabel('pose residence')
    ylabel('mean norm rate')
    
    set([h(1) h(2)],'xscale','log')
    hl = legend(uarea);
    
    for ia=1:numel(uarea)
        s = hl.String{ia};
        hl.String{ia} = sprintf('%s: R=%.3g, p=%.3g',s,rg(ia),pg(ia));
    end
    
    % save
    sname = [figdir '/corr_poseResid_rate.pdf'];
    save2pdf(sname,gcf)
end



%% metrics for correspondence of rate and embedding maps
if getMapMetrics
    % NB: maybe spread out the energy for modules
    
    fprintf('map metrics')
    tic
    
    % init
    K = [];
    Kr = [];
    K_mod = [];
    Kr_mod = [];
    
    DI_embed = [];
    DI_rate = [];
    DI_rand = [];
    DI_embed_mod = [];
    DI_rate_mod = [];
    DI_rand_mod = [];
    
    % loop
    for ii=1:numel(res_emb_rate)
        dotdotdot(ii,0.1,numel(res_emb_rate))
        
        % extract
        id = SDF(ii).id;
        
        p1 = res_emb_rate(ii).dens_rate;
        p2 = res_emb_rate(ii).dens_embed;
        
        p1 = p1 ./ nansum(p1(:));
        p2 = p2 ./ nansum(p2(:));
              
        % kl, pose
        K(ii,1) = kldiv_nd(p1,p2);

        % kl, modules
        % - spreadt energy to compare with pose
        if spreadEnergy
            pm = pose2mod(id,:);
            p1m = p1;
            p2m = p2;
            for ic=1:max(pm)
                sel = ismember(cluster_train.Ld,find(pm==ic));
                p1m(sel) = nanmean(p1m(sel));
                p2m(sel) = nanmean(p2m(sel));
            end
            K_mod(ii,1) = kldiv_nd(p1m,p2m);
        else
            % prep for module remapping
            oldval = 1:nstate;
            newval = pose2mod(id,:);
            tmpld = cluster_train.Ld;
            tmpld_mod = changem(tmpld,newval,oldval);
            tmpld_mod(tmpld_mod==0) = max(newval)+1;
            tmpld_mod = [tmpld_mod(:)];

            p1m = accumarray(tmpld_mod,[p1(:)]);
            p1m(end) = [];
            p1m = p1m ./ sum(p1m);

            p2m = accumarray(tmpld_mod,[p2(:)]);
            p2m(end) = [];
            p2m = p2m ./ sum(p2m);

            K_mod(ii,1) = kldiv(1:numel(p1m),p1m',p2m');
        end
        
        % density-based, poses
        for ic=1:nstate
            sel = cluster_train.Ld==ic;
            d1 = nansum(p1(sel));
            d2 = nansum(p2(sel));
            DI_rate(ii,ic) = d1;
            DI_embed(ii,ic) = d2;
        end
        
        if 0
            % density-based, modules
            for ic=1:max(pose2mod(id,:))
                tmpc = pose2mod(id,:);
                tmpc = find(tmpc==ic);
                sel = ismember(cluster_train.Ld,tmpc);
                d1 = nansum(p1(sel));
                d2 = nansum(p2(sel));
                DI_rate_mod(ii,ic) = d1;
                DI_embed_mod(ii,ic) = d2;
            end
        end
        
        % rand?
        if isfield(res_emb_rate,'dens_rate_rand') && ~isempty(res_emb_rate(ii).dens_rate_rand)
            p3 = res_emb_rate(ii).dens_rate_rand;
            p3 = p3 ./ sum(sum(p3,1),2);
            tmpp = reshape(p3,[size(p3,1)*size(p3,2) size(p3,3)]);

            % kl
            for ir=1:size(p3,3)
                Kr(ii,ir) = kldiv_nd(p3(:,:,ir),p2);
            end
            
            for ir=1:size(p3,3)
                if spreadEnergy
                    p3m = p3(:,:,ir);
                    pm = pose2mod(id,:);
                    for ic=1:max(pm)
                        sel = ismember(cluster_train.Ld,find(pm==ic));
                        p3m(sel) = nanmean(p3m(sel));
                    end
                    Kr_mod(ii,ir) = kldiv_nd(p3m,p2m);
                else
                    p3m = accumarray(tmpld_mod,tmpp(:,ir));
                    p3m(end) = [];
                    p3m = p3m ./ sum(p3m);
                    Kr_mod(ii,ir) = kldiv(1:numel(p1m),p3m',p2m');
                end
            end
            
            % density-based
            for ic=1:nstate
                sel = tmpld==ic;
                dr = nansum(tmpp(sel,:));
                DI_rand(ii,ic,:) = dr;
            end   
            
            if 1
                % density-based, modules
                for ic=1:max(pose2mod(id,:))
                    sel = ismember(tmpld_mod,id);
                    dr = nansum(tmpp(sel,:));
                    DI_rand_mod(ii,ic,:) = dr;
                end
            end
        end
        
        % clean
        if 0
        is = max(pose2mod(id,:))+1;
        DI_embed_mod(ii,is:end) = nan;
        DI_rate_mod(ii,is:end) = nan;
        DI_rand_mod(ii,is:end,:) = nan;
        end
    end

    % extract map metrics
    Kr_mu = nanmean(Kr,2);
    Kr_mod_mu = nanmean(Kr_mod,2);
    Ki = (K - Kr_mu) ./ (K + Kr_mu);

    toc
end


%% plot dissimialrity 
if plotMapsSimilarity
    if 0
        % get stuff
        DS = cat(3,res_emb_rate.D);
        DSn = cat(3,res_emb_rate.Dn);
        DS_rand = permute(cat(4,res_emb_rate.D_rand),[1 2 4 3]);

        % get mean between and within
        fprintf('compiling within/between distance')
        dW = [];
        dB = [];
        dWr = [];
        dBr = [];
        for ii=1:size(DS,3)
            dotdotdot(ii,0.1,size(DS,3))

            ds = DS(:,:,ii);
            dn = DSn(:,:,ii);
            dsr = squeeze(DS_rand(:,:,ii,:));

            % between selection
            selb = triu(ones(size(ds)),1);
            selb = selb & ~(isnan(ds) | ds==0);

            % within
            dw = diag(ds);
            dW(ii,1) = nanmean(dw);

            % between
            nb = dn(selb);
            db = ds(selb);
            dB(ii,1) = sum(nb.*db) ./ sum(nb);

            % random
            for ir=1:size(dsr,3)
                dWr(ii,ir) = nanmean(diag(dsr(:,:,ir)));
                tmp = dsr(:,:,ir);
                tmp = tmp(selb);
                dBr(ii,ir) = sum(nb.*tmp) ./ sum(nb);
            end
        end
        
        % finish
        dWr = nanmean(dWr,2);
        dBr = nanmean(dBr,2);
    end
    
    % plot mean within and between
    figure
    nr = 1; nc = 3; ns=0;
    set_bigfig(gcf,[0.8 0.3])
    doBar = 0;
    
    % plot all within
    [mur,ser] = avganderror_group(iarea,dWr,'mean',200);
    
    ns = ns+1;
    subplot(nr,nc,ns)
    cfg = [];
    cfg.dobar = doBar;
    cfg.groups = uarea;
    cfg.ylabel = 'mean within dissimalrity';
    cfg.title = 'within-pose dissimalrity';
    out1 = plot_anova(dW,iarea,cfg);  
    hold all
    hb = errorbar(out1.xx,mur,ser,'k.-','markersize',10);
    legend([out1.hbar,hb],{'obs','rand'},'location','southeast')
    
    % plot all between
    [mur,ser] = avganderror_group(iarea,dBr,'mean',200);

    ns = ns+1;
    subplot(nr,nc,ns)
    cfg = [];
    cfg.dobar = doBar;
    cfg.groups = uarea;
    cfg.ylabel = 'mean between dissimalrity';
    cfg.title = 'between-pose dissimalrity';
    out2 = plot_anova(dB,iarea,cfg);  
    hold all
    errorbar(out1.xx,mur,ser,'k.-','markersize',10);
    legend([out2.hbar,hb],{'obs','rand'},'location','southeast')

    % plot ratio
    D = (dB - dW) ./ (dW + dB);
    Dr = (dBr - dWr) ./ (dWr + dBr);
    [mur,ser] = avganderror_group(iarea,Dr,'mean',200);

    ns = ns+1;
    subplot(nr,nc,ns)
    cfg = [];
    cfg.dobar = doBar;
    cfg.groups = uarea;
    cfg.ylabel = 'mean diff';
    cfg.title = 'diff in dissimialrity (=between-within/between+within)';
    out3 = plot_anova(D,iarea,cfg);
    hold all
    errorbar(out1.xx,mur,ser,'k.-','markersize',10);
    legend([out3.hbar,hb],{'obs','rand'},'location','southeast')

    % save
    sname = [figdir '/map_similarity'];
    save2pdf(sname,gcf)
    
end

%% plot density differences by pose/actions
if plotMapsDensities
    % prep
    tmp = cluster_train.Ld(:);
    n = accumarray(tmp(tmp>0),1,[nstate 1])';
    
    di_rate = DI_rate ./ n;
    di_embed = DI_embed ./ n;
    
    d = di_embed - di_rate;
    d = d*100;
    
    % means
    [mu,se] = avganderror_group(iarea,d,'mean');
    
    % start figure
    figure
    [nr,nc] = subplot_ratio(numel(uarea)+1,2);
    set_bigfig(gcf,[0.35 0.3])
    ns = 0;
    hax = [];
    
    % mean diff per pose
    ns=ns+1;
    subplot(nr,nc,ns)
    hp = plot(mu');
    plotcueline('y',0)
    legend(hp,uarea,'location','eastoutside')
    title('density diff (=embed-rate) per area')
    xlabel('pose')
    ylabel('mean density diff')
    hax(ns) = gca;
    axis square
    
    % plot diffs per area
    lim = [];
    for ia=1:numel(uarea)
        sel = iarea==ia;
        tmp1 = cat(3,res_emb_rate(sel).dens_rate);
        tmp2 = cat(3,res_emb_rate(sel).dens_embed);
        d = tmp2 - tmp1;
        d = nanmean(d,3)*100;
        lim(ia,:) = [min(d(:)) max(d(:))];
        
        % rate embedding
        ns=ns+1;
        subplot(nr,nc,ns)
        tmpc = cluster_train;
        tmpc.outClust.dens2 = d;        
        out = plot_embedding(tmpc.outClust,tmpc.Lbnds,tmpc.Ld,0,0);
        title(uarea{ia})
        hax(ns) = gca;
    end
    colormap(french)
    equalbounds(hax(2:end),'clim')
    setaxesparameter(hax(2:end),'clim',get(gca,'clim')/2)
    
    % save
    sname = [figdir '/map_density_diff'];
    save2pdf(sname,gcf)
end

%% plot mean KL dievregnce between maps
if plotMapsKL
    
    rateBins = [0 0.1 0.5 1 Inf];
    
    % extract
    Kr2 = nanmean(Kr,2);    
    rate = [SDF.mean_rate];
    
    % start figure
    figure
    [nr,nc] = subplot_ratio(numel(rateBins));
    ns=0;
    set_bigfig(gcf,[0.3 0.4]);
    
    for ir=1:numel(rateBins)
        if ir < numel(rateBins)
            selr = rate >= rateBins(ir) & rate <= rateBins(ir+1);
            rstr = sprintf('rate=%g-%g',rateBins(ir),rateBins(ir+1));
        else
            selr = true(size(rate));
            rstr = 'all';
        end
        
        k = K(selr);
        g = iarea(selr);
        kr = Kr2(selr);
        [mur,ser] = avganderror_group(g,kr,'mean');
        
        % plot for pose
        ns = ns+1;
        subplot(nr,nc,ns)
        cfg = [];
        cfg.groups = uarea(unique(g));
        cfg.ugroups = unique(g);
        cfg.ylabel = 'mean KL div';
        cfg.title = sprintf('mean KL div, %s',rstr);
        out1 = plot_anova(k,g,cfg);  
        hold all
        hp=errorbar(out1.xx,mur,ser,'k.-','markersize',10);

        legend([out1.hbar hp],{'obs','rand'},'location','northwest')

        axis square
        set(gca,'xtick',xx,'xticklabel',uarea);
    end
        
    % save
    sname = [figdir '/embed_maps_kldiv.pdf'];
    save2pdf(sname,gcf)
end


%% plot density-based metrics
if plotMapsKL_modularity
    %K2 = (K-Kr_mu) ./ (K+Kr_mu);
    %K_mod2 = (K_mod-Kr_mod_mu) ./ (K_mod+Kr_mod_mu);
    K2 = K ./ Kr_mu;
    K_mod2 = K_mod ./ Kr_mod_mu;
    dK = K_mod2 - K2;
    
    % start figure
    figure
    nr = 1; nc = 3;ns=0;
    set_bigfig(gcf,[0.4 0.2])
    
    % plot for pose
    ns = ns+1;
    subplot(nr,nc,ns)
    cfg = [];
    cfg.groups = uarea;
    cfg.ylabel = 'mean (norm) KL div';
    cfg.title = 'rate vs embed divergence: pose';
    out1 = plot_anova(K2,iarea,cfg);  
    
    % plot for modularity
    ns = ns+1;
    subplot(nr,nc,ns)
    cfg = [];
    cfg.groups = uarea;
    cfg.ylabel = 'mean (norm) KL div';
    cfg.title = 'rate vs embed divergence: module';
    out2 = plot_anova(K_mod2,iarea,cfg);  
    
    % plot difference
    ns = ns+1;
    subplot(nr,nc,ns)
    cfg = [];
    cfg.groups = uarea;
    cfg.ylabel = 'mean (norm) KL div diff (=mod-pose)';
    cfg.title = 'module divergence - pose divergence';
    out3 = plot_anova(dK,iarea,cfg); 
    
    %finihs
    set([out1.ax,out2.ax,out3.ax],'fontsize',6)
    %setaxesparameter('ylim')
end


%% plot all example maps
if plotMaps_examples
    isVis = 0;
    
    exdir = [figdir '/example_maps'];
    if ~exist(exdir); mkdir(exdir); end
    
    fprintf('plotting example maps')
    for ii=1:numel(res_emb_rate)
        dotdotdot(ii,0.1,numel(res_emb_rate));
        
        % prep
        k = res_emb_rate(ii).kl;
        kr = res_emb_rate(ii).kl_rand;
        pe = res_emb_rate(ii).dens_embed;
        pr = res_emb_rate(ii).dens_rate;
        prr = nanmean(res_emb_rate(ii).dens_rate_rand,3);
            
        tmpc = cluster_train;
                
        % start figure
        tmpstr={'off','on'};
        figure('visible',tmpstr{isVis+1})
        nr = 2; nc = 3; ns=0;
        set_bigfig(gcf,[0.35 0.3])
        hax = [];
        
        % embedding
        ns=ns+1;
        subplot(nr,nc,ns)
        tmpc.outClust.dens2 = pe;        
        out = plot_embedding(tmpc.outClust,tmpc.Lbnds,tmpc.Ld,0);
        title('pose embedding')
        hax(ns) = out.hax;
        
        % rate embedding
        ns=ns+1;
        subplot(nr,nc,ns)
        tmpc.outClust.dens2 = pr;        
        out = plot_embedding(tmpc.outClust,tmpc.Lbnds,tmpc.Ld,0);
        title('rate-weighted pose embedding')
        hax(ns) = out.hax;

        % rand rate
        ns=ns+1;
        subplot(nr,nc,ns)
        tmpc.outClust.dens2 = prr;        
        out = plot_embedding(tmpc.outClust,tmpc.Lbnds,tmpc.Ld,0);
        title('rand rate-weighted pose embedding')
        hax(ns) = out.hax;

        % pose - rate embeddding
        ns=ns+1;
        subplot(nr,nc,ns)
        tmpc.outClust.dens2 = pe-pr;        
        out = plot_embedding(tmpc.outClust,tmpc.Lbnds,tmpc.Ld,0,0);
        title('diff(pose-rate) embeddings')
        hax(ns) = out.hax;
        equalbounds('clim')
        colormap(gca,french)

        % KL divergence
        ns=ns+1;
        subplot(nr,nc,ns)
        hh = histogram(kr,10);
        hp=plotcueline('x',k,'linewidth',2);
        title('dissimilarity of pose and rate-weighted pose maps')
        xlabel('KL div (pose vs rate map')
        legend([hh hp],{'rand','obs'})

        % finish
        setaxesparameter(hax(1:3),'clim')
        
        % save
        name = res_emb_rate(ii).name(1:end-19);
        a = areas{ii};
        sname = sprintf('%s/%s_%s_exMap.pdf',exdir,a,name);
        save2pdf(sname)
        
        foo=1;
    end
end

%% mean density maps
if plotMapsMean
    % ------------------------------------------------------
    % plot maps
    figure;
    nr = 3; nc = numel(uarea);
    hax =[];
    set_bigfig(gcf,[1 0.8])
    
    for ia=1:numel(uarea)
        sela = iarea==ia;
        dens_rate = cat(3,res_emb_rate(sela).dens_rate);
        dens_embed = cat(3,res_emb_rate(sela).dens_embed);
        
        % plot mean embedding map
        mu = nanmean(dens_embed,3);
        tmp = cluster_train;
        tmp.outClust.dens2 = mu;
        
        ns=ia;%+numel(uarea)
        subplot(nr,nc,ns)
        %imagesc(mu)
        plot_embedding(tmp.outClust,tmp.Lbnds,tmp.Ld,0);
        
        axis square
        colorbar
        title(['embedding: ' uarea{ia}])
        hax(1,ia)=gca;
        
        % plot mean rate map
        mu = nanmean(dens_rate,3);
        tmp = cluster_train;
        tmp.outClust.dens2 = mu;
        
        ns=ia+numel(uarea);
        subplot(nr,nc,ns)
        %imagesc(mu)
        plot_embedding(tmp.outClust,tmp.Lbnds,tmp.Ld,0);
        
        axis square
        colorbar
        title(['rate: ' uarea{ia}])
        hax(2,ia)=gca;

        % plot mean differences
        mu = nanmean(dens_embed-dens_rate,3);
        tmp = cluster_train;
        tmp.outClust.dens2 = mu;
        
        ns=ia+2*numel(uarea);
        subplot(nr,nc,ns)
        %imagesc(mu)
        plot_embedding(tmp.outClust,tmp.Lbnds,tmp.Ld,0);
        
        axis square
        colorbar
        %colormap(gca,french)
        %equalbounds(gca,'clim')
        title(['embed-rate: ' uarea{ia}])
        hax(3,ia)=gca;
    end
    setaxesparameter(hax(1:2,:),'clim')
    setaxesparameter(hax(3,:),'clim')
    tightfig
    
    % save
    sname = [figdir '/embed_maps_means.pdf'];
    save2pdf(sname,gcf)
    
end


%% correlate rate vs KL
if plotMapsKLvsRate
    R = [SDF.mean_rate];
    
    figure
    hs = scatterhist(K,R,'group',areas,'kernel','on','marker','.','markersize',10);
    set(hs(1),'xscale','log','yscale','log')
    set(hs(2),'xscale','log')
    set(hs(3),'xscale','log')
    
    xlabel('rate')
    ylabel('KL div')
    
    % save
    sname = [figdir '/embed_maps_corr_rate.pdf'];
    save2pdf(sname,gcf)
end




%% rate on embedding
if plotEmbeddedRate_pose
    minRate = 0.05;

    % prep
    figdir2 = [figdir '/example_embedded_rate'];
    if ~exist(figdir2); mkdir(figdir2); end

    
    % what to plot?
    %meanRates = cellfun(@nanmean,{SDF.sdf});
    meanRates = [SDF.mean_rate];
    thesePlots = find(meanRates > minRate);

    % loop
    fprintf('figures, embedded rate')
    for itmp=1:numel(thesePlots)
        ii = thesePlots(itmp);
        dotdotdot(itmp,0.1,numel(thesePlots))
        id = SDF(ii).id;
        
        % spike info
        f1 = SDF(ii).frame;
        sdf = SDF(ii).sdf;
        
        % pose info
        seldat = idat==id & ismember(frame,f1);
        f2 = frame(seldat);
        y = Y_test(seldat,:);

        % bin rate
        nbin = 30;
        xv = cluster_train.outClust.xv;
        yv = cluster_train.outClust.yv;
        
        d = mean(diff(xv));
        xe = linspace(xv(1),xv(end)+d,nbin+1); 
        ye = linspace(yv(1),yv(end)+d,nbin+1); 
        [N,~,~,ibinr,ibinc] = histcounts2(y(:,2),y(:,1),ye,xe);
        
        xx = nanmean([xe(1:end-1); xe(2:end)]);
        yy = nanmean([ye(1:end-1); ye(2:end)]);
        
        r = accumarray([ibinr,ibinc],sdf,size(N));
        r = r ./ N;
        
        % smooth
        K = 2;
        r = interpn(r,K);
        xx = interpn(xx,K);
        yy = interpn(yy,K);
        
        % plot
        figure('visible','off')
        
        imagesc(xx,yy,r)
        hold all
        [icol,irow] = find(cluster_train.Lbnds==1);
        hbnd = plot(xv(irow),yv(icol),'k.');
        
        axis square
        set(gca,'clim',[0 max(get(gca,'clim'))])
        hc = colorbar;
        hc.Title.String = 'mean rate';

        s = sprintf('%s\n%s, ch=%g',datasets(id).name,a{ii},SDF(ii).ch);
        title(s)
        xlabel('umap 1')
        ylabel('umap 2')
        
        % save
        sname = sprintf('%s/%s_%s',figdir2,datasets(id).name(1:end-9),a{ii});
        save2pdf(sname)
        close(gcf)
        
        foo=1;
    end
    
end

%% behavioural scores vs neural activity
if plotCorrNeuralVSbehav
    vars = {'kl','q'};
    corrtype = 'spearman';

    % extract
    ids = [SDF.id];
    dgs = DGS(:,ids)';
    %dgs = nanmean(DGSr(:,ids,:),3)';
    tds = TDS(:,ids)';
    q = Q(:,ids)';
    if any(strcmp(vars,'stab'))
        [stab,stabr]=get_stability(Qlab,Qrlab,lags,idat,nrand,nstate);
        stab = [nan(numel(SDF),1), stab(:,ids)'];
    end
    
    kl = K - nanmean(Kr,2);
    %kl = K;
    kl = repmat(kl,1,size(dgs,2));
    
    rsa = repmat(RSA,1,size(dgs,2));
    
    % start figure
    figure
    nr = 1; nc = 2;
    set_bigfig(gcf,[0.3 0.2]);
    
    % plots
    uarea2 = [{'all'}; uarea];
    cols = get_safe_colors(0);
    cols = [zeros(1,3); cols];
    h=[];
    for ia=1:numel(uarea)+1
        if ia == 1
            sel = true(size(iarea));
            ns = 1;
        else
            sel = iarea==ia-1;
            ns = 2;
        end
        
        str = sprintf('x=%s;',vars{1});
        eval(str);
        str = sprintf('y=%s;',vars{2});
        eval(str);
        
        % correlate DGS with KL diveregnce
        [r,p] = corr(x(sel,:),y(sel,:),'type',corrtype);
        r = r(1,:)';
        p = p(1,:);
        %r = r(:,1)';
        %p = p(:,1);
        r = smooth(r,10);
        
        mup = ones(size(r)) * max(r)*1.05;

        % plot
        subplot(nr,nc,ns)
        
        h(ia) = plot(trans_lags,r,'color',cols(ia,:),'linewidth',2);
        set(gca,'xscale','log')
        plotcueline('y',0)

        str = sprintf('corr %s w %s',vars{1},vars{2});
        title(str)
        xlabel('transition lags')
        ylabel([corrtype ' R'])
        axis square
    end
    legend(h(2:end),uarea,'location','southeast')
    
end

%% neural dimensioanlity vs behavioural stability
if plotNeuralDimVSbehavStab
    
    warning('OFF', 'stats:pca:ColRankDefX')
    
    % settings
    thresh = 90;
    
    figdir2 = [figdir '/Figure_stability'];
    if ~exist(figdir2); mkdir(figdir2); end

    
    % loop over datasets
    if calcStab
        out_stab = [];

        ndat = numel(datasets);
        fprintf('neural stability')
        for id=1:numel(datasets)
            dotdotdot(id,0.1,numel(datasets));
        
            name = datasets(id).name;
            day = datestr( name(4:end-19) );
            seln = ismember({SDF.day},day);

            sdf = SDF(seln);

            % concatenate all datasets
            uframe = unique([sdf(:).frame]);

            X = nan(numel(uframe),numel(sdf));
            for is=1:numel(sdf)
                x = sdf(is).sdf;
            
                % reduce outlier influence
                if 0
                bad = abs(x) > 50;
                idx=1:numel(x);
                xi = interp1(idx(~bad),x(~bad),find(bad),'pchip','extrap');
                x(bad) = xi;
                %x = medfilt1(x,15);
                end

                % store
                [~,self] = ismember(uframe,sdf(is).frame);
                X(self,is) = x;
            end
            
            % smooth
            X2 = medfilt1(X,5);
            
            % normalize        
            %X = X ./ max(X);
            X = zscore(X);

            % estimate dimensianlity: PCA
            [coeff,score,latent,tsquared,v_exp,mu] = pca(X,'Centered',0,'Economy',0);
            [~, dimThresh] = min(cumsum(v_exp) > thresh == 0);

            % direction change in PCA space
            if 0
                a = score(1:end-1,:);
                b = score(2:end,:);

                %d = sum(a./b,2) ./ (sqrt(sum(a.^2,2)).*sqrt(sum(b.^2,2))); % cosine sim
                d = sqrt(sum((a-b).^2,2));

                mu = median(d);
                se = mad(d,1)*1.4826*6;
                bad = d<mu-se | d > mu+se;
                idx=1:numel(d);
                di = interp1(idx(~bad),x(~bad),find(bad),'pchip','extrap');
                d(bad) = di;
            end
            
            % estimate behavioural stability
            c = C(idat==id);
            istate = find( diff(c)~=0 );
            n_trans = numel(istate);
            istate_time = zeros(size(c));
            istate_time(istate+1) = 1;


            % store
            tmp = [];
            tmp.var_exp = v_exp;
            tmp.neural_dim90 = dimThresh;
            tmp.n_trans = n_trans';
            tmp.n_neurons = size(X,2);
            tmp.id = id;
            tmp.area = [sdf.area];
            tmp.score = score;
            tmp.coeff = coeff;
            tmp.n_trans_time = istate_time;

            out_stab = cat(1,out_stab,tmp);


            foo=1;
        end
        fprintf('\n')
        
        % save it for later
        if 0
            fprintf('\n saving... \n')
            sname = [figdir2 '/out_stab.mat'];
            save(sname,'out_stab');
        end
    end

    
    foo=1;
    
    % correlate areal difference with transitions
    if 0
        maxLag = 1000;

        dwin = ceil(0.1*fs_frame);
        win = gausswin(dwin);
        win = win ./ sum(win);

        Rc = [];
        Rc_lag = [];
        Rc_area = [];
        Rc_lag_area = [];
        DC_area = {};
        DC = [];
        
        fprintf('xcorr')
        for id=1:numel(out_stab)
            dotdotdot(id,0.1,numel(out_stab))
            
            score = out_stab(id).score;
            coef = out_stab(id).coeff;
            ar = collapse_areas(out_stab(id).area);
            v = out_stab(id).var_exp;
            
            % clean
            for is=1:size(score,2)
                s = score(:,is);
            
                %mu = median(s);
                %se = mad(s,1)*1.4826*6;
                %bad = s<mu-se | s > mu+se;
                bad = abs(s) > 100;
                idx=1:numel(s);
                si = interp1(idx(~bad),s(~bad),find(bad),'pchip','extrap');
                s(bad) = si;
                n = conv(ones(size(s)),win,'same');
                s = conv(s,win,'same') ./ n;
                
            end
            
            % get state change
            a = score(1:end-1,:);
            b = score(2:end,:);

            if 1 % euclidean
                dc = abs(a-b);
                d = sqrt(sum(dc.^2,2));
            else % cosine sim
                d = sum(a./b,2) ./ (sqrt(sum(a.^2,2)).*sqrt(sum(b.^2,2))); 
                d = abs(d);
            end
            d = [0; d];
            
            % change by component
            coef = coef ./ max(abs(coef));
            dc_scale = dc ./ max(dc);
            dc_scale = dc_scale * coef';
            dc_scale = dc_scale .* v';
            dc_scale = [zeros(1,size(dc_scale,2)); dc_scale];
            
            DC_area = [DC_area,ar];
            DC = [DC, mean(dc_scale)];

            % get transition times
            t = out_stab(id).n_trans_time;
            t = conv(t,win,'same');

            % cross correlate overall change
            [c,lags] = xcorr(t,d,maxLag,'coeff');
            [~,imx] = max(c);
            Rc(id) = c(imx);
            Rc_lag(id) = lags(imx);
            
            % cross correlate change by area
            for ic=1:size(dc_scale,2)
                tmpd = dc_scale(:,ic);
                [c,lags] = xcorr(t,tmpd,maxLag,'coeff');
                [~,imx] = max(c);
                
                iR = numel(Rc_area)+1;
                Rc_area(iR) = c(imx);
                Rc_lag_area(iR) = lags(imx);
            end
            foo=1;
        end
        fprintf('\n')
    end
    
    % correlate with shorter windows of time
    if 0
        % settings
        maxLag = 1000;

        dwin = ceil(0.1*fs_frame);
        win = gausswin(dwin);
        win = win ./ sum(win);

        win_xcorr = ceil(10*fs_frame); % samples;
        
        % loop over datasets
        Rc_time = [];
        Rc_lag_time = [];
        
        fprintf('xcorr')
        for id=1:numel(out_stab)
            dotdotdot(id,0.1,numel(out_stab))
            
            score = out_stab(id).score;
            coef = out_stab(id).coeff;
            ar = collapse_areas(out_stab(id).area);
            v = out_stab(id).var_exp;
            
            % get transition times
            t = out_stab(id).n_trans_time;
            t = conv(t,win,'same');
            
            % xcorr over shorter windows
            binEdge = 1:win_xcorr:numel(t);
            for ib=1:numel(binEdge)-1
                tmpd = score(:,1);
                
                % cross correlate overall change
                [c,lags] = xcorr(t,d,maxLag,'coeff');
                [~,imx] = max(c);
                Rc_time(id) = c(imx);
                Rc_time_lag(id) = lags(imx);
            end

        end
        
        
    end
    
    % ---------------------------------
    % plot xcorr by area
    
    figure
    nr = 1; nc = 2;
    [G,~,iG] = unique(DC_area);
    xx = 1:numel(G);

    % max corr
    mu = grpstats(Rc_area, iG, {@mean});
    se = grpstats(Rc_area, iG,{@(x) nanstd(bootstrp(200,@(x) mean(x),x))});
    [p, atab] = anovan(Rc_area,iG,'display','off');
    
    
    subplot(nr,nc,1)
    barwitherr(se,xx,mu)
    
    str = sprintf('max cross-corr per area\n anova F=%.3g, p=%.3g',atab{2,6},p);
    title(str)
    ylabel('max cross-corr')
    
    axis square
    set(gca,'xtick',xx,'xticklabel',G);
    
    % max lag
    tmp_lag = Rc_lag_area ./ fs_frame;
    mu = grpstats(tmp_lag, iG, {@mean});
    se = grpstats(tmp_lag, iG,{@(x) nanstd(bootstrp(200,@(x) mean(x),x))});
    [p, atab] = anovan(tmp_lag,iG,'display','off');
    
    
    subplot(nr,nc,2)
    barwitherr(se,xx,mu)
    
    str = sprintf('max lag per area\n anova F=%.3g, p=%.3g',atab{2,6},p);
    title(str)
    ylabel('max lag (pos=neural precedes pose transition')
    
    axis square
    set(gca,'xtick',xx,'xticklabel',G);
    
    
    % ---------------------------------
    % plot change point by area
    [G,~,iG] = unique(DC_area);
    mu = grpstats(DC, iG, {@mean});
    se = grpstats(DC, iG,{@(x) nanstd(bootstrp(200,@(x) mean(x),x))});
    [p, atab] = anovan(DC',iG,'display','off');
    
    xx = 1:numel(G);
    
    figure
    barwitherr(se,xx,mu)
    
    str = sprintf('mean influence of each area to state change\n anova F=%.3g, p=%.3g',atab{2,6},p);
    title(str)
    ylabel('mean influence to state change')
    
    axis square
    set(gca,'xtick',xx,'xticklabel',G);
    
    % ---------------------------------
    % plot xcorr stuff
    figure
    nr = 1; nc = 2;
    
    % max correlation
    [mu,se] = avganderror(Rc,'mean');
    
    subplot(nr,nc,1)
    hist(Rc,20)
    
    plotcueline('x',0)
    plotcueline('x',mu,'r-')    

    tstr = sprintf('cross corr of pose transitions and neural trajectory change\nmean R=%.3g+%.3g, p=%.3g',...
        mu,se,p);
    title(tstr)
    xlabel('R')
    ylabel('freq')
    
    
    % lags
    tmp_lag = Rc_lag ./ fs_frame;
    [mu,se] = avganderror(tmp_lag,'mean');
    
    subplot(nr,nc,2)
    hist(tmp_lag,20)
    
    plotcueline('x',0)
    plotcueline('x',mu,'r-')
    

    tstr = sprintf('mean lag=%.3g+%.3g',...
        mu,se);
    title(tstr)
    xlabel('lag in sec (pos=neural traj change predicts pose transitions)')
    ylabel('freq')
    
    % ------------------------------------------------------
    % variance explained by area
    %   Each column of COEFF contains coefficients for one principa component.
    ar = {};
    A = [];
    ic = 1;
    for id=1:numel(out_stab)
        a = out_stab(id).area;
        a = collapse_areas(a);
        
        c = out_stab(id).coeff(:,ic);
        %c = c ./ max(c(:));
        %c = abs(c);
        c = c - min(c);
        c = c ./ max(c(:));
        v = out_stab(id).var_exp(ic);
        
        % append
        ar = [ar,a];
        %A = [A,c'*v];
        A = [A,c'];
        foo=1;
    end
    
    % plot per area
    [G,~,iG] = unique(ar);
    mu = grpstats(A, iG, {@mean});
    se = grpstats(A, iG,{@(x) nanstd(bootstrp(200,@(x) mean(x),x))});
    [p, atab] = anovan(A',iG,'display','off');
    
    xx = 1:numel(G);
    
    figure
    barwitherr(se,xx,mu)
    
    str = sprintf('mean abs infleunce of area on 1st PC \nanova F=%.3g, p=%.3g',atab{2,6},p);
    title(str)
    ylabel('mean abs influence')
    
    axis square
    set(gca,'xtick',xx,'xticklabel',G);
    
    foo=1;
end


%% neural dim vs behav stability in time
if plotNeuralDimVSbehavStab_time
    warning('OFF', 'stats:pca:ColRankDefX')
    
    % settings
    thresh = 90;
    win = ceil(30*30); % samples
    %step = ceil(win/4); % samples
    anaWin = 10*60*30; %10
    splitByArea = 1;

    
    stabdir = sprintf('%s/neur_behav_stab_time%g',sdfpath,anaWin);
    figdir2 = sprintf('%s/neur_behav_stab_time%g',figdir,anaWin);

    
    if ~exist(figdir2); mkdir(figdir2); end
    if ~exist(stabdir); mkdir(stabdir); end

    % calculate stability
    if calcStabTime
        
        if nparallel > 0 && isempty(gcp('nocreate'))
            myPool = parpool(nparallel);
        end
        
        tic
        
        % loop over datasets
        fprintf('calculating stability stuff...\n')
        if splitByArea
            area_idx = 1:numel(uarea);
        else
            area_idx = 1;
        end
        
        %for id=numel(datasets)
        parfor id=1:numel(datasets)

            name = datasets(id).name;
            day = datestr( name(4:end-19) );
            fprintf('%g: %s\n',id,name)
            
            % select
            for ia=area_idx
                thisArea = uarea{ia};
                seln = ismember({SDF.day},day);

                if splitByArea
                    selsdf = seln & ismember(areas',thisArea);
                    areastr = uarea{ia};
                else
                    selsdf = seln;
                    areastr = 'all';
                end
                sdf = SDF(selsdf);

                if numel(sdf)<2
                    continue
                end

                % concatenate all datasets
                uframe = unique([sdf(:).frame]);

                X = nan(numel(uframe),numel(sdf));
                for is=1:numel(sdf)
                    x = sdf(is).sdf;

                    % reduce outlier influence
                    bad = abs(x) > 50;
                    idx=1:numel(x);
                    xi = interp1(idx(~bad),x(~bad),find(bad),'pchip','extrap');
                    x(bad) = xi;
                    %x = medfilt1(x,15);

                    % store
                    [~,self] = ismember(uframe,sdf(is).frame);
                    X(self,is) = x;
                end

                % normalize        
                %X = X ./ max(X);
                %X = (X-nanmean(X)) ./ nanstd(X);
                X = (X-nanmedian(X)) ./ (mad(X,1)*1.4826);

                % estimate behavioural stability
                c = C(idat==id);
                istate = find( diff(c)~=0 );
                tmp = zeros(size(c));
                tmp(istate) = 1;
                ntrans_time = tmp;

                % calculate for multiple time windows
                if anaWin > 0
                    step = ceil(anaWin/2);
                    st = 1:step:numel(c)-step;
                    fn = anaWin:step:numel(c);
                    if numel(st)>numel(fn); st(end) = []; end                
                    binEdge = [st;fn]';
                else
                    binEdge = [1 numel(c)];
                end

                v_exp = nan(0,128);
                d = [];
                n_trans = [];
                score = [];
                coeff = [];
                r = [];
                for ib=1:size(binEdge,1)
                    st = binEdge(ib,1);
                    fn = binEdge(ib,2);

                    % neural dimensionality
                    %   Rows of SCORE correspond to observations, columns to components.
                    %   Each column of COEFF contains coefficients for one principa component.
                    x = X(st:fn,:);
                    [tmpco,s,latent,tsquared,v,mu] = pca(x,'Economy',0);
                    %[tmpco,s,latent,tsquared,v,mu] = pca(x,'Centered',0,'Economy',0);

                    [~, dimThresh] = min(cumsum(v) > thresh == 0);

                    d(ib) = dimThresh;
                    v_exp(ib,1:numel(v)) = v;
                    %score(:,:,ib) = s;
                    %coeff(:,:,ib) = tmpco;
                    coeff(:,ib) = tmpco(:,1);

                    % behav stability
                    n_trans(ib) = sum(ntrans_time(st:fn)) ./ (fn-st);

                    % rate
                    r(:,ib) = nanmean(x);

                    foo=1;
                end

                % store
                tmp = [];
                tmp.var_exp = v_exp;
                tmp.neural_dim90 = d';
                tmp.n_trans = n_trans';
                tmp.n_neurons = size(X,2);
                tmp.id = id;
                tmp.score = score;
                tmp.coeff = coeff;
                tmp.area = areas(selsdf);
                tmp.rate = r;

                % save partial
                sname = [stabdir '/' name(1:end-9) '_' areastr '_stab.mat'];
                parsave(sname,tmp)
            end
            
            foo=1;
        end
        
        toc
       
        % load it back
        fprintf('loading stability stuff')
        
        if splitByArea
            area_idx = 1:numel(uarea);
        else
            area_idx = 1;
        end

        out_stab_time = [];
        isBad = false(numel(datasets),numel(area_idx));
        for id=1:numel(datasets)
            dotdotdot(id,0.1,numel(datasets));
            name = datasets(id).name;
            
            % area split?
            for ia=1:numel(uarea)
                if splitByArea
                    areastr = uarea{ia};
                else
                    areastr = 'all';
                end
                sname = [stabdir '/' name(1:end-9) '_' areastr '_stab.mat'];

                if exist(sname)
                    in = load(sname);
                else
                    isBad(id,ia) = 1;
                    continue
                end

                if numel(out_stab_time)==0
                    out_stab_time = repmat(in,numel(datasets),numel(area_idx));
                else
                    out_stab_time(id,ia) = in;
                end
            end
        end
        
        % clean if there was no data
        for ii=1:numel(out_stab_time)
            ff = fields(out_stab_time);
            if isBad(ii)
                for jj=1:numel(ff)
                    tmp = out_stab_time(ii).(ff{jj});
                    if iscell(tmp)
                        tmp = repmat({'na'},size(tmp));
                    else
                        tmp = nan(size(tmp));
                    end
                    out_stab_time(ii).(ff{jj}) = tmp;
                end
            end
            out_stab_time(ii).isBad = isBad(ii);
        end
    end
    
    foo=1;
    
    % ------------------------------------------------------------
    % if session wide, calculate correlation
    if anaWin==0 && splitByArea==0
        corrtype = 'spearman';
       
        % start plot
        figure
        nr = 2; nc = 3; ns=0;
        set_bigfig(gcf,[0.7 0.7])
        
        % mean var explained
        v=cat(1,out_stab_time.var_exp);
        v = v(:,1:50);
        [mu,se] = avganderror(v,'mean');
        
        ns=ns+1;
        subplot(nr,nc,ns)
        barwitherr(se,mu)
        title('mean var explained')
        xlabel('PC')
        ylabel('mean var explained')
        axis square
        
        % variance in first PC
        v=cat(1,out_stab_time.var_exp);
        v=v(:,1);
        n=[out_stab_time.n_trans]';
        bad = isnan(v);
        [r,p] = corr(v(~bad),n(~bad),'type',corrtype);
        
        ns=ns+1;
        subplot(nr,nc,ns)
        scatter(v,n)
        s = sprintf('corr transition probability vs var exp in PC1\n%s R=%.3g, p=%.3g',...
            corrtype,r,p);
        title(s)
        xlabel('var explained in 1st PC')
        ylabel('number of transitions')
        axis square
        
        % corr across all PC
        r=[];
        p=[];
        for ii=1:50
            v=cat(1,out_stab_time.var_exp);
            v=v(:,ii);
            n=[out_stab_time.n_trans]';
            bad = isnan(v);
            [r(ii),p(ii)] = corr(v(~bad),n(~bad),'type',corrtype);
        end
        mup = ones(size(p)) * max(r)*1.05;
        mup(p>0.05) = nan;
        
        ns=ns+1;
        subplot(nr,nc,ns)
        bar(r)
        hold all
        plot(1:numel(mup),mup,'k.','markersize',10)
        title('correlation across diff PCs')
        xlabel('PC')
        ylabel([corrtype ' R'])
        axis square
        
        % plot coefficient score per area
        c = cat(1,out_stab_time.coeff);
        c = abs(c);
        a = cat(1,out_stab_time.area);
        [ua,~,ia] = unique(a);
        
        [mu,se] = avganderror_group(ia,c,'median');
        
        ns=ns+1;
        subplot(nr,nc,ns)
        errorbar(1:numel(mu),mu,se)
        set(gca,'xtick',1:numel(mu),'xticklabel',ua)
        title('areal influence on var exp')
        ylabel('mean abs(coeff)')
        
        % correlate coefficient score with p(trans)
        c = cat(1,out_stab_time.coeff);
        c = abs(c);
        v=cat(1,out_stab_time.var_exp);
        v = v(:,1);
        v = cellfun(@(x,y) repmat(x,size(y)),num2cell(v),{out_stab_time.area}','un',0);
        v = cat(1,v{:});
        c = c.*v ./ sum(v);
        
        a = cat(1,out_stab_time.area);
        [ua,~,ia] = unique(a);
        
        n = cellfun(@(x,y) repmat(x,size(y)),num2cell([out_stab_time.n_trans]),{out_stab_time.area},'un',0);
        n = cat(1,n{:});
        
        [r,p,ugroup]=corr_group(ia,c,n,'type',corrtype);
        mup = ones(size(r)) * max(r) * 1.05;
        mup(p>0.05) = nan;
        
        ns=ns+1;
        subplot(nr,nc,ns)
        bar(r)
        hold all
        plot(1:numel(mup),mup,'k.','markersize',10)
        set(gca,'xtick',1:numel(r),'xticklabel',ua)
        title('corr P(trans) vs areal coeff')
        ylabel([corrtype ' R'])
    end
    
    
    % ------------------------------------------------------------
    % calculate correlation for each day
    if anaWin>0 && splitByArea==0
        corrtype = 'spearman';

        % start figure
        figure
        nr = 2; nc = 3; ns = 0;
        set_bigfig(gcf,[0.7 0.7])
        
        % mean variance explained
        v = [];
        for id=1:numel(out_stab_time)
            v(id,:)= nanmean(out_stab_time(id).var_exp);
        end
        v = v(:,1:50);
        [mu,se] = avganderror(v,'mean');
        xx = 1:size(v,2);
        
        ns=ns+1;
        subplot(nr,nc,ns)
        shadedErrorBar(xx,mu,se)
        title('mean var explained')
        xlabel('PC')
        ylabel('mean var explained')
        axis square
                
        % mean correlation var exp vs p(trans) per session
        R = [];
        P = [];
        for id=1:numel(out_stab_time)
            v = out_stab_time(id).var_exp(:,1);
            b = out_stab_time(id).n_trans(:,1);

            [r,p] = corr(v,b,'type',corrtype);
            R(id) = r;
            P(id) = p;
        end
        [mu,se] = avganderror(R,'mean');
        psig = sum(P<0.05);

        ns = ns+1;
        subplot(nr,nc,ns)
        hist(R,20)
        plotcueline('x',0)
        plotcueline('x',mu,'r-')
        
        tstr = sprintf('corr P(trans) and var exp in PC1\nmean %s R=%.3g+%.3g, psig=%g/%g',...
            corrtype,mu,se,psig,numel(P));
        title(tstr)
        xlabel([corrtype ' R'])
        ylabel('freq')
        axis square
        
         % mean correlation ndim=90% vs p(trans) per session
        R = [];
        P = [];
        for id=1:numel(out_stab_time)
            v = out_stab_time(id).neural_dim90;
            v = v ./ out_stab_time(id).n_neurons; % normalize
            b = out_stab_time(id).n_trans(:,1);

            [r,p] = corr(v,b,'type',corrtype);
            R(id) = r;
            P(id) = p;
        end
        [mu,se] = avganderror(R,'mean');
        psig = sum(P<0.05);

        ns = ns+1;
        subplot(nr,nc,ns)
        hist(R,20)
        plotcueline('x',0)
        plotcueline('x',mu,'r-')
        
        tstr = sprintf('corr P(trans) and ndim(90) in PC1\nmean %s R=%.3g+%.3g, psig=%g/%g',...
            corrtype,mu,se,psig,numel(P));
        title(tstr)
        xlabel([corrtype ' R'])
        ylabel('freq')
        axis square
        
        % mean correlation per session, diff PC
        R = [];
        P = [];
        for id=1:numel(out_stab_time)
            for ip=1:50
                if ip > size(out_stab_time(id).var_exp,2)
                    continue
                end
                v = out_stab_time(id).var_exp(:,ip);
                b = out_stab_time(id).n_trans;

                [r,p] = corr(v,b,'type',corrtype);
                R(id,ip) = r;
                P(id,ip) = p;
            end
        end
        [mu,se] = avganderror(R,'mean');
        psig = sum(P<0.05);
        xx = 1:numel(mu);
        %mup = ones(size(mu)) * max(mu) * 1.05;
        %mup(P>0.05) = nan;
        
        ns = ns+1;
        subplot(nr,nc,ns)
        shadedErrorBar(xx,mu,se)
        hold all
        %plot(xx,mup,'r.','markersize')
        plotcueline('y',0)
        
        tstr = sprintf('corr across diff PC');
        title(tstr)
        xlabel('PC')
        ylabel(['mean ' corrtype ' R'])
        axis square
        
        % mean areal-influence
        c = cellfun(@(x) x(:),{out_stab_time.coeff},'un',0);
        c = cat(1,c{:});
        c = abs(c);
        
        a = cellfun(@(x,y) repmat(x,1,size(y,2)),{out_stab_time.area},{out_stab_time.coeff},'un',0);
        a = cellfun(@(x) x(:),a,'un',0);
        a = cat(1,a{:});
        [ua,~,ia] = unique(a);
        
        [mu,se] = avganderror_group(ia,c,'mean');
        
        ns=ns+1;
        subplot(nr,nc,ns)
        
        errorbar(1:numel(ua),mu,se)
        set(gca,'xtick',1:numel(mu),'xticklabel',ua)
        title('areal influence on var exp')
        ylabel('mean abs(coeff)')
        
        % correlate areal-influence with trans prob
        r = nan(numel(out_stab_time),128);
        a = cell(size(r));
        for id=1:numel(out_stab_time)
            for ip=1:size(out_stab_time(id).coeff,1)
                tmpc = out_stab_time(id).coeff(ip,:)';
                tmpc = abs(tmpc);
                p = out_stab_time(id).n_trans;
                
                r(id,ip) = corr(tmpc,p,'type',corrtype);
                a(id,ip) = out_stab_time(id).area(ip);
            end
        end
        r = r(:,1:50);
        a = a(:,1:50);
        bad = isnan(r);
        r(bad) = [];
        a(bad) = [];
        
        [ua,~,ia] = unique(a);

        [mu,se] = avganderror_group(ia,r,'mean');
        
        ns=ns+1;
        subplot(nr,nc,ns)
        
        errorbar(1:numel(ua),mu,se)
        set(gca,'xtick',1:numel(mu),'xticklabel',ua)
        title('corr coeff with P(trans)')
        ylabel('mean R')
        axis square
    end
    
    
    % ------------------------------------------------------------
    % variance explained by each area
    if anaWin > 1 && splitByArea==1        
        % start figure
        figure
        nr = 2; nc = 3; ns=0;
        cols = get_safe_colors(0,1:numel(uarea));
        
        % mean variance explained
        v = [];
        for id=1:size(out_stab_time,1)
            for ia=1:numel(uarea)
                v(:,ia,id)= nanmean(out_stab_time(id,ia).var_exp);
            end
        end
        v = v(1:50,:,:);
        [mu,se] = avganderror(v,'mean',3);
        xx = 1:size(v,1);
        
        ns=ns+1;
        subplot(nr,nc,ns)
        h = [];
        for ia=1:numel(uarea)
            htmp = shadedErrorBar(xx,mu(:,ia),se(:,ia),{'color',cols(ia,:)});
            h(ia) = htmp.mainLine;
            hold all
        end
        title('mean var explained')
        xlabel('PC')
        ylabel('mean var explained')
        axis square
        legend(h,uarea,'location','northeast')
        
         % mean correlation var exp vs p(trans) per session
        R = [];
        P = [];
        for id=1:size(out_stab_time,1)
            for ia=1:numel(uarea)
                v = out_stab_time(id,ia).var_exp(:,1);
                b = out_stab_time(id,ia).n_trans(:,1);

                [r,p] = corr(v,b,'type',corrtype);
                R(id,ia) = r;
                P(id,ia) = p;
            end
        end
        R = abs(R);
        [mu,se] = avganderror(R,'mean');
        psig = sum(P<0.05);
        
        ns = ns+1;
        subplot(nr,nc,ns)
        errorbar(1:numel(uarea),mu,se)
        set(gca,'xtick',1:numel(uarea),'xticklabel',uarea)
        
    end
    
    % ------------------------------------------------------------
    % corss correlate
    if 0
        maxLag = 1000;
        nrand = 100;

        Rc = [];
        Rc_lag = [];
        Rc_r = [];
        Rc_lag_r = [];
        for id=1:numel(out_stab_time)
            v = out_stab_time(id).var_exp(:,1);
            b = out_stab_time(id).n_trans(:,1);

            [c,lags] = xcorr(b,v,maxLag,'coeff');
            [~,imx] = max(c);
            Rc(id) = c(imx);
            Rc_lag(id) = lags(imx);

            for ir=1:nrand
                idx = randi(ceil(numel(v)/2));
                vr = circshift(v,idx);

                [cr,lagsr] = xcorr(b,vr,maxLag,'coeff');
                [~,imx] = max(cr);
                Rc_r(id,ir) = cr(imx);
                Rc_lag_r(id,ir) = lagsr(imx);
            end

            foo=1;
        end

        figure
        nr = 1; nc = 2;

        % max correlation
        [mu,se] = avganderror(Rc,'mean');
        tmpr = mean(Rc_r);
        p = sum(abs(tmpr) > mu) ./ nrand;

        subplot(nr,nc,1)
        hist(Rc,20)

        plotcueline('x',0)
        plotcueline('x',mu,'r-')    

        tstr = sprintf('cross corr of behav stability and neural dimensioality\nmean R=%.3g+%.3g, p=%.3g',...
            mu,se,p);
        title(tstr)
        xlabel('R')
        ylabel('freq')


        % lags
        [mu,se] = avganderror(Rc_lag,'mean');
        tmpr = mean(Rc_lag_r);
        p = sum(abs(tmpr) > mu) ./ nrand;

        subplot(nr,nc,2)
        hist(Rc_lag,20)

        plotcueline('x',0)
        plotcueline('x',mu,'r-')


        tstr = sprintf('cross corr of behav stability and neural dimensioality\nmean lag=%.3g+%.3g, anaWin=%g sec',...
            mu,se,anaWin./30);
        title(tstr)
        xlabel('lag (neg=neural var predicts behav var)')
        ylabel('freq')

        foo=1;
    end
end

%% pose tuning
if plotPoseTuning
    [out_pose,hout_pose] = get_tuning('pose',SDF,idat,C);

    sname = [sdfpath '/out_pose.mat'];
    save(sname,'out_pose')

    sname = [figdir '/tuning_pose'];
    save2pdf(sname,hout_pose.fig_tuning)
    sname = [figdir '/tuning_pose_byArea'];
    save2pdf(sname,hout_pose.fig_area)
end

%% plot pose specificity
if plotSpecificity_pose
    
    % settings
    threshs = [0.1:0.01:0.9];
    
    % extract
    m = out_pose.MU;
    mr = out_pose.MUr;
    
    % get thresholded specificty
    d = nanmean(m - mr,3);
    
    S = [];
    for it=1:numel(threshs)
        th = max(d,[],2) * threshs(it);
        s = sum(d > th,2);
        good=s>0;
        n = accumarray(s(good),1,[nstate 1]);
        n = n ./ sum(n);
        
        S(:,it) = n;
    end
    
    %ratio of max and to others
    d2 = d - min(d,[],2);
    d2(isnan(d2)) = 0;
    mx = max(d2,[],2);
    r = d2 ./ mx;
    r = sort(r,2,'descend');
    
    % plot
    figure
    nr = 1; nc = 3;
    set_bigfig(gcf,[0.8 0.3])
    
    % plot per threshold
    xx = threshs;
    yy = 1:nstate;
    [~,imx] = max(S);
    mx = yy(imx);
    ylim = [1 15];
    
    ns = 1;
    subplot(nr,nc,ns)
    imagesc(xx,yy,S)
    hold all
    plot(threshs,mx,'k-','linewidth',2)
    
    axis square
    set(gca,'ydir','normal','ylim',ylim)
    colorbar
    
    title('proportion of cells that meet threshold')
    xlabel('selectivity thresh (rel to max)')
    ylabel('n state that meet criteria')
    
    % same plot, but different
    ncol = 10;
    S2 = S(1:ncol-1,:);
    S2(ncol,:) = sum(S(ncol:end,:));
    
    ns = ns+1;
    subplot(nr,nc,ns)
    
    hb = bar(threshs,S2,'stacked');
    
    cmap = jet(ncol);
    for ib=1:ncol
        set(hb(ib),'facecolor',cmap(ib,:),'edgecolor','none','barwidth',1)
    end
    lstr = cellfun(@num2str,num2cell(1:ncol),'un',0);
    lstr{end} = [lstr{end} '+'];
    hl=legend(hb(1:ncol),lstr,'location','eastoutside');
    hl.Title.String = 'n sel pose';
    
    axis square
    set(gca,'ylim',[0 1],'xlim',[threshs(1) threshs(end)])
    
    title('proprotion of neurons exhibiting selectivity')
    xlabel('selectivity thresh (rel to max)')
    ylabel('proportion')
    
    % plot mean ratio
    [mu,se] = avganderror(r,'mean');
    
    ns = ns+1;
    subplot(nr,nc,ns)
    shadedErrorBar(yy,mu,se,{'k.-','markersize',10})
    
    axis square
    title('mean relative rate')
    xlabel('sorted state')
    ylabel('relative rate to max')
   
end

%% pose specifity by area
if plotSpecificity_pose_byArea
    % settings
    threshs = [0.1:0.01:0.9];
    
    % extract
    m = out_pose.MU;
    mr = out_pose.MUr;
    
    % get thresholded specificty
    d = nanmean(m - mr,3);
    
    % plot selectivity per area
    Sa = [];
    for ia=1:numel(uarea)
        for it=1:numel(threshs)
            sel = strcmp(areas,uarea{ia});
            d2 = d(sel,:);
            
            th = max(d2,[],2) * threshs(it);
            s = sum(d2 > th,2);
            good=s>0;
            n = accumarray(s(good),1,[nstate 1]);
            n = n ./ sum(n);

            Sa(:,it,ia) = n;
        end
    end
    
    % select a threshold
    nstate_sel = 10;
    ith = nearest(threshs,0.3);
    sa = squeeze(Sa(1:nstate_sel,ith,:));
    
    figure
    hp=plot(sa');
    set(hp,{'Color'}, num2cell(copper(nstate_sel),2))

    set(gca,'xtick',1:numel(uarea),'xticklabel',uarea)
    
    % save
    sname = [figdir '/specificity_pose_byArea'];
    save2pdf(sname,gcf)
end


%% module tuning
if plotModuleTuning
    [out_module,hout_module] = get_tuning('module',SDF,idat,C_mod);

    sname = [figdir '/out_module.mat'];
    save(sname,'out_module')

    sname = [figdir '/tuning_module'];
    save2pdf(sname,hout_module.fig_tuning)
    sname = [figdir '/tuning_module_byArea'];
    save2pdf(sname,hout_module.fig_area)

end

%% pose transition

if plotTransitionTuning_pose
    % CAREFUL: info bleeds because of windowing, and between datasets

    % only select pre-transitions
    tlim = -[0.1 0]; %sec
    tlim = round(tlim * fs_frame);

    st = [0; find(diff(C_mod)~=0)]+1;
    fn = [st(2:end); numel(C_mod)]-1;
    tmpc = nan(size(C));
    for is=2:numel(st)
        %s = st(is) + tlim(1);
        %f = fn(is) + tlim(2);
        s = fn(is)+tlim(1);
        f = fn(is)+tlim(2);
        s = max(s,fn(is-1)-round(timwin(1)*fs_frame));
        f = f-round(timwin(2)*fs_frame);
        tmpc(s:f) = C(f+round(timwin(2)*fs_frame)+1);
    end
    C_pose_trans = tmpc;

    [out_pose_trans,hout_pose_trans] = get_tuning('pose trans',SDF,idat,C_pose_trans);

    sname = [figdir '/out_pose_trans.mat'];
    save(sname,'out_pose_trans')

    sname = [figdir '/tuning_pose_trans'];
    save2pdf(sname,hout_pose_trans.fig_tuning)
    sname = [figdir '/tuning_pose_trans_byArea'];
    save2pdf(sname,hout_pose_trans.fig_area)
end

%% module transition
if plotTransitionTuning_module
    % CAREFUL: info bleeds because of windowing, and between datasets

    % only select pre-transitions
    tlim = [-0.1 0]; %sec
    tlim = round(tlim * fs_frame);

    st = [0; find(diff(C_mod)~=0)]+1;
    fn = [st(2:end); numel(C_mod)]-1;
    %st(1) = [];
    %fn(1) = [];
    tmpc = nan(size(C_mod));
    for is=2:numel(st)
        %s = st(is) + tlim(1);
        %f = fn(is) + tlim(2);
        s = fn(is)+tlim(1);
        f = fn(is)+tlim(2);
        s = max(s,fn(is-1)-round(timwin(1)*fs_frame));
        f = f-round(timwin(2)*fs_frame);
        tmpc(s:f) = C_mod(f+round(timwin(2)*fs_frame)+1);
    end
    C_mod_trans = tmpc;

    [out_module_trans,hout_module_trans] = get_tuning('module trans',SDF,idat,C_mod_trans);
    
    sname = [figdir '/out_module_trans.mat'];
    save(sname,'out_module_trans')

    sname = [figdir '/tuning_module_trans'];
    save2pdf(sname,hout_module_trans.fig_tuning)
    sname = [figdir '/tuning_module_trans_byArea'];
    save2pdf(sname,hout_module_trans.fig_area)
end

foo=1;

%% 
if plotPeriTransition_module
    tlim = [-1 1];
    tlim = ceil(tlim * fs_frame);
    
    st = [0; find(diff(C_mod)~=0)]+1;
    fn = [st(2:end); numel(C_mod)]-1;
    
    % get rate for each cell
    udat = unique([SDF.id]);
    R = [];
    for id=1:numel(udat)
        dotdotdot(id,0.1,numel(udat))
        sel = idat==udat(id);
        %sel = ~isnan(C) & C~=0;

        c = C_mod_control(sel);
        st = find(diff(c)~=0)+1;

        % get rate per pose
        id_spk = find([SDF.id]==udat(id));

        for is=1:numel(id_spk)
            is2 = id_spk(is);
            for ic=1:numel(SDF(is2).label)
                f = SDF(is2).frame;
                r = SDF(is2).sdf(ic,:);
                
                mu = nanmedian(r);
                se = mad(r,1)*1.4286;
                r = (r - mu) ./ se;
                
                Rtmp = [];
                for ist=1:numel(st)
                    s = max(st(ist)+tlim(1),1);
                    f = min(st(ist)+tlim(2),numel(r));
                    
                    idx = tlim(1):tlim(2);
                    idx2 = s:f;
                    
                    good = ismember(idx,idx2-st(ist));
                    tmp = nan(1,diff(tlim)+1);
                    tmp(good) = r(s:f);

                    Rtmp(ist,:) = tmp;
                    foo=1;
                end
                
                im = size(R,1)+1;
                R(im,:) = nanmedian(Rtmp);
                
                foo=1;
            end
        end
    end
    
    % ------------------------------------------
    % overall
    
    % plot
    t = tlim(1):tlim(2);
    [mu,se] = avganderror(R,'median',1,1,200);
    
    figure
    shadedErrorBar(t,mu,se)
    xlabel('time from module transition')
    ylabel('norm SDF')
    
    % ------------------------------------------
    % per area
    % areas
    iG = iarea;
    
    if 1
        mu = grpstats(R, iG, {@median});
        se = grpstats(R, iG,{@(x) nanstd(bootstrp(200,@(x) median(x),x))});
    else
        mu = grpstats(R, iG, {@mean});
        se = grpstats(R, iG,{@(x) nanstd(bootstrp(200,@(x) mean(x),x))});
    end
    
    figure
    [nr,nc] = subplot_ratio(numel(uarea));
    set_bigfig(gcf,[0.8 0.8])
    
    for ii=1:numel(uarea)
        m = mu(ii,:);
        s = se(ii,:);
        
        subplot(nr,nc,ii)
        shadedErrorBar(t,m,s)
        
        str = sprintf('%s',uarea{ii});
        title(str)
        xlabel('time from module transition')
        ylabel('norm SDF')
        axis square
    end
    
    foo=1;
end

%% rate for transition control
if plotControlRate_module
    tstrs = {'pre','mid','post'};
    
    tlim = [-0.3 0; 0 0.3]; %sec
    tlim = round(tlim * fs_frame);

    st = [0; find(diff(C_mod)~=0)]+1;
    fn = [st(2:end); numel(C_mod)]-1;

    tmpc = ones(size(C_mod))*2;
    for is=2:numel(st)-1
        % before
        s = fn(is)+tlim(1,1);
        f = fn(is)+tlim(1,2);
        s = max(s,fn(is-1));
        tmpc(s:f) = 1;
        
        % after
        s = fn(is)+tlim(2,1);
        f = fn(is)+tlim(2,2);
        f = min([f,fn(is+1),numel(tmpc)]);
        tmpc(s:f) = 3;
    end
    C_mod_control = tmpc;
    
    % get rate for each cell
    udat = unique([SDF.id]);
    R = [];
    for id=1:numel(udat)
        dotdotdot(id,0.1,numel(udat))
        sel = idat==udat(id);
        %sel = ~isnan(C) & C~=0;

        c = C_mod_control(sel);
        uc = unique(c);

        % get rate per pose
        id_spk = find([SDF.id]==udat(id));

        for is=1:numel(id_spk)
            is2 = id_spk(is);
            for ic=1:numel(SDF(is2).label)
                r = SDF(is2).sdf(ic,:);
                f = SDF(is2).frame;
                %a = repmat(RES(is2).area(ic),size(r));

                im = size(R,1)+1;

                for ir=1:numel(uc)
                    sel = c==uc(ir);
                    R(im,ir) = nanmean(r(sel));
                end
            end
        end
    end
    
    % ----------------------------------------------------
    % overall
    
    % normalize
    R2 = R ./ max(abs(R),[],2);
    
    % means
    [mu,se] = avganderror(R2,'median',1,1,200);
    xx = 1:numel(uc);
    
    % plot
    figure
    barwitherr(se,xx,mu)
    set(gca,'xtick',xx,'xticklabel',tstrs)
    
    % ----------------------------------------------------
    % analysis per area
    
    % areas
    iG = iarea;
    
    if 1
        mu = grpstats(R, iG, {@median});
        se = grpstats(R, iG,{@(x) nanstd(bootstrp(200,@(x) median(x),x))});

    else
        mu = grpstats(R, iG, {@mean});
        se = grpstats(R, iG,{@(x) nanstd(bootstrp(200,@(x) mean(x),x))});
    end
    
    % plot
    fig2 = figure();
    xx = 1:numel(uarea);
    
    if 0
        hb = barwitherr(se,xx,mu);
        set(hb,'facecolor',ones(1,3)*0.8)
    else
        errorbar(xx,mu,se)
    end
    set(gca,'xtick',xx,'xticklabel',uarea)
    xtickangle(30)
    axis square
    
    
    foo=1;
end


%% control tuning
if plotControlTuning_module
    tlim = [-0.1 0; 0 0.1]; %sec
    tlim = round(tlim * fs_frame);

    st = [0; find(diff(C_mod)~=0)]+1;
    fn = [st(2:end); numel(C_mod)]-1;

    tmpc = nan(size(C_mod));
    for is=2:numel(st)-1
        % before
        s = fn(is)+tlim(1,1);
        f = fn(is)+tlim(1,2);
        s = max(s,fn(is-1));
        %s = max(s,fn(is-1)-round(timwin(1)*fs_frame));
        %f = f-round(timwin(2)*fs_frame);
        tmpc(s:f) = 1;
        
        % after
        s = fn(is)+tlim(2,1);
        f = fn(is)+tlim(2,2);
        f = min([f,fn(is+1),numel(tmpc)]);
        %s = max(s,fn(is-1)-round(timwin(1)*fs_frame));
        %f = f-round(timwin(2)*fs_frame);
        tmpc(s:f) = 2;
    end
    C_mod_control = tmpc;
    
    
    [out_module_control,hout_module_control] = get_tuning('module control',SDF,idat,C_mod_control);
    
    sname = [figdir '/out_module_control.mat'];
    save(sname,'out_module_control')

    sname = [figdir '/tuning_module_control'];
    save2pdf(sname,hout_module_control.fig_tuning)
    sname = [figdir '/tuning_module_control_byArea'];
    save2pdf(sname,hout_module_control.fig_area)
    
    
%     % get rates
%     nrand = 20;
%     %udat = unique(idat);
%     udat = unique([SDF.id]);
%     nstate = max(C);
%     
%     for id=1:numel(udat)
%         dotdotdot(id,0.1,numel(udat))
%         sel = idat==udat(id);
%         %sel = ~isnan(C) & C~=0;
% 
%         c = C(sel);
%             
%             
%     end
end

%% full example of action tuning
if plotExampleTuning_module
    iG = iarea;
    
    % select
    thisLabel = C;
    f = out_module.F' - nanmean(out_module.Fr,2);
    
    sel = find(strcmp(areas,'ACC') & f > 0.01);
    
    for is=1:numel(sel)
        is2 = sel(is);
        id = SDF(is2).id;
        ch = SDF(is2).ch;
        icell = SDF(is2).label;
        
        % load in spk data
        name = datasets(id).name(1:end-9);
        tmp = [spkparentpath '/' name];
        tmpd = dir([tmp '/*nt' num2str(ch) 'ch*']);
        load(tmpd(1).name);
        
        % convert to sec
        for ic=1:numel(spk.time)
            spk.time{ic} = spk.time{ic} ./ fs_spk;
        end
        spk.trialtime = spk.trialtime ./ fs_spk;

        
        % ---------------------------
        figure
        nr = 2; nc = 1;
        hax = [];
        
        % sdf
        cfg = [];
        %cfg.timwin = [-0.25 0.25];
        cfg.timwin = [-0.1 0.1];
        cfg.latency = [0 180000/30];
        sdf = ft_spikedensity(cfg,spk);
        
        subplot(nr,nc,1)
        plot(sdf.time,sdf.avg)
        xlabel('time')
        ylabel('SDF')
        hax(1) = gca;
        
        % raster
        %subplot(nr,nc,2)
        %plot(sdf.time,ones(size(sdf.time)),'k.','markersize',10)
        %hax(2) = gca;

        % plot states
        seld = idat==id;
        c = thisLabel(seld);
        f = frame(seld);
        f = f ./ fs_frame;
        
        [uc,~,ic] = unique(c);
        
        st = [0; find(diff(c)~=0)]+1;
        fn = [st(2:end); numel(c)]-1;
    
        subplot(nr,nc,2)
        imagesc(ic')
        %plot(f,ic)
        
        hax(2) = gca;
        
        % finish
        setaxesparameter(hax,'xlim',[0 600])
        

        for ic=1:numel(uc)
            selc = c==uc(ic);
            y = ones(sum(selc))*ic;
            f2 = f(selc);
            plot(f2,y,'.')
            hold all
        end
        
        
    end
    
end

%% example of tuning of neurons
if exampleTuning



end

%% are neurons more tuned to pose or module?
if plotHierarchyTuning
    % prep
    if 0
        T_pose = out_pose.F' ./ mean(out_pose.Fr,2);
        T_mod = out_module.F' ./ mean(out_module.Fr,2);
    elseif 0
        T_pose = out_pose_trans.F' ./ mean(out_pose_trans.Fr,2);
        T_mod = out_module_trans.F' ./ mean(out_module_trans.Fr,2);
    else
        T_pose = out_pose_trans.F';
        T_mod = out_module_trans.F';
    end

    %iHier = (T_mod - T_pose) ./ (T_mod + T_pose);
    iHier = T_mod - T_pose;
    %iHier = T_mod ./ T_pose - 1;

    iG = iarea;
    
    % stats
    mu = grpstats(iHier, iG, {@mean});
    se = grpstats(iHier, iG,{@(x) nanstd(bootstrp(200,@(x) mean(x),x))});

    G = {iG};
    [p,t] = anovan(iHier,G,'display','off');

    % plot
    figure('name','module vs pose')
    xx = 1:numel(uarea);
    hb = barwitherr(se,xx,mu);
    set(hb,'edgecolor','none')

    set(hb,'facecolor',ones(1,3)*0.8)
    set(gca,'xtick',xx,'xticklabel',uarea)
    xtickangle(30)
    axis square

    str = sprintf('Norm Hierarchy tuning (=Tmodule - Tpose)\nANOVAN F=%.3g, p=%.3g',t{2,6},p);
    title(str)
    ylabel('Mean Tuning')

    set_bigfig(gcf,[0.3 0.35])

    sname = [figdir '/tuning_poseVSmodule'];
    save2pdf(sname,gcf)
end

%% prep for video
if prepSampleVideo
    if 0
        T = out_module.F' ./ mean(out_module.Fr,2);
        theseID = find(out_module.R' > 1 & T>10);
    else
        id = 5;
        %seld = [SDF.id]'==id & T_mod>12; % & out_module.R' > 1
        seld = find([SDF.id]'==id);
        theseID = find(seld);
        theseID=625;
    end
    %theseID = 421;

    tmp_id = [SDF.id];
    tmp_ch = [SDF.ch];
    for id=1:numel(theseID)
        id2 = theseID(id);

        s = SDF(id2).sdf;
        f = SDF(id2).frame;
        name = datasets(SDF(id2).id).name(1:end-9);
        ch = tmp_ch(id2);

        [foo,~] = fileparts(anadir);

        viddir = [foo '/' name '/viz'];
        savedir = [figdir '/spike_vid'];

        fprintf('================\nexample vid: %s, ch=%g\n',name,ch)
        spike_audio_sdf(s,f,name,ch,viddir,savedir)
    end
    %sdf = SDF(
end





%% =================================================================
% ==================================================================
% ==================================================================
% MISC
% ==================================================================
% ==================================================================
% ==================================================================


function [out,hout] = get_tuning(varargin)

    % inputs
    if isstruct(varargin{1}) % already run
        out = varargin{1};
        ctype = out.ctype;
        SDF = varargin{2};
        idat = varargin{3};
        C = varargin{4};

        getStats = 0;
    else
        ctype=varargin{1};
        SDF=varargin{2};
        idat=varargin{3};
        C=varargin{4};

        getStats = 1;
    end

    doAnova = 0; 
    doMI_continuous = 0; 
    doMI_discreet = 1;
        nbin = 20;
    
    K_mi = 2;
    nrand = 20;
    %udat = unique(idat);
    udat = unique([SDF.id]);
    nstate = max(C);

    % get stats?
    if getStats
        tic
        nRandSeg = 10;

        MU = [];
        MUr = [];
        P = [];
        Pr = [];
        R = [];
        F = [];
        Fr = [];

        fprintf('%s tuning',ctype)
        for id=1:numel(udat)
            dotdotdot(id,0.1,numel(udat))
            sel = idat==udat(id);
            %sel = ~isnan(C) & C~=0;

            tmpc = C(sel);
            tmpc(isnan(tmpc)) = -1;
            [uc,~,c] = unique(tmpc);
            if uc(1) == -1
                uc(1) = [];
                c(tmpc==-1) = nan;
                c = c-1;
            end
            
            %if any(uc==0)
            %    c = c-1;
            %end
            
            % get rate per pose
            [uc,~,ic] = unique(c(~isnan(c)));
            nc = accumarray(ic,1);
            notEnough = nc <= K_mi;
            
            %selgood = c > 0 & ~ismember(c,uc(notEnough));
            selgood = c > 0 & ~isnan(c) & ~ismember(c,uc(notEnough));
            npose = accumarray(c(selgood),1,[nstate 1]);

            id_spk = find([SDF.id]==udat(id));

            for is=1:numel(id_spk)
                is2 = id_spk(is);
                for ic=1:numel(SDF(is2).label)
                    r = SDF(is2).sdf(ic,:);
                    f = SDF(is2).frame;
                    %a = repmat(RES(is2).area(ic),size(r));

                    rate = mean(r);

                    % norm rate
                    %r = zscore(r);
                    %r=log(r+1);
                    %r=real(r);
                    

                    %[B,DEV,STATS] = glmfit(...)
                    %[~,~,tmpc] = unique(c);
                    %[B,dev,stats] = mnrfit(r,c);
                    %mdl = fitglm(r,tmpc,'pose ~ rate','varnames',{'rate','pose'},'distribution','binomial');

                    if doAnova
                        %[p,t] = kruskalwallis(r(selgood),c(selgood),'off');
                        [p,t] = anova1(r(selgood),c(selgood),'off');
                        tmpf = t{2,5};
                    elseif doMI_continuous
                        tmpf = mi_discrete_cont(r(selgood)',c(selgood)',K_mi);
                        p = nan;
                    elseif doMI_discreet
                        % discretize
                        %rbin = linspace(0,100,1000);
                        rbin = nbin;
                        [~,~,ibin] = histcounts(r,rbin);
                        ibin(ibin==0) = 1;
                        r = ibin';
                        
                        tmpf = mi(r(selgood),c(selgood));
                        p = nan;
                    end

                    % mean rate per pose
                    n = accumarray(c(selgood),r(selgood),[nstate 1]);
                    mu = n ./ npose;

                    im = size(MU,1)+1;
                    MU(im,:) = mu;
                    P(im) = p;
                    R(im) = rate;
                    F(im) = tmpf;

                    % randomize
                    nsmp = numel(f);
                    tmpmu = nan(numel(npose),nrand);
                    tmpp = nan(1,nrand);
                    tmpf = nan(1,nrand);
                    %parfor ir=1:nrand
                    for ir=1:nrand
                        if 0
                            idx = randperm(nsmp);
                            rr = r(idx);
                        else
                            st = randi(nsmp);
                            rr = circshift(r,st);
                        end

                        nr = accumarray(c(selgood),rr(selgood),[nstate 1]);
                        mu = nr ./ npose;
                        tmpmu(:,ir) = mu;

                        if doAnova
                            %[p,t] = kruskalwallis(rr,c,'off');
                            [p,t] = anova1(rr(selgood),c(selgood),'off');

                            tmpf(ir) = t{2,5};
                            tmpp(ir) = p;
                        elseif doMI_continuous
                            tmpf(ir) = mi_discrete_cont(rr(selgood)',c(selgood)',K_mi);
                            tmpp(ir) = nan;
                        elseif doMI_discreet
                            tmpf(ir) = mi(rr(selgood),c(selgood));
                            tmpp(ir) = nan;
                        end
                    end
                    % store
                    MUr(im,:,:) = tmpmu;
                    Pr(im,:) = tmpp;
                    Fr(im,:) = tmpf;
                    foo=1;
                end
            end

            foo=1;
        end
        toc
    else
        out = [];
        out.ctype = ctype;
        MU = out.MU;
        MUr = out.MUr;
        P = out.P;
        Pr = out.Pr;
        F = out.F;
        Fr = out.Fr;
    end

    % summary
    nsig = sum(P<0.05);
    fprintf('%g/%g, %.3g %% cells rate predicts %s\n',nsig,numel(P),100*nsig./numel(P),ctype);

    % ------------------------------------------------------------
    % plot
    p = sum(Fr>F',2)./nrand;

    fig1 = figure('name',[ctype ' tuning']);
    hbo = histogram(F,20,'normalization','probability');
    hold all
    hbr = histogram(nanmean(Fr,2),20,'normalization','probability');

    set([hbo hbr],'edgecolor','none')

    str = sprintf('pose tuning\nnsig=%.3g/%.3g',sum(p<0.05),numel(F));
    title(str)
    xlabel('X2')
    ylabel('cell freq')

    legend([hbo hbr],{'observed','random'})



    % ------------------------------------------------------------
    % do analysis per area

    % tuning index
    %T = F' ./ nanmean(Fr,2);
    T = F' - nanmean(Fr,2);

    % areas
    a = [SDF.area]';
    a = collapse_areas(a);

    % cuz dumb shit
    %a(strcmp(a,'SMA')) = {'PM'};

    [uarea,~,iG] = unique(a);

    if 0
        mu = grpstats(T, iG, {@median});
        se = grpstats(T, iG,{@(x) nanstd(bootstrp(200,@(x) median(x),x))});

        G = {a};
        [~,~,iF] = unique(T');
        [p,t] = anovan(iF,G,'display','off');
    else
        mu = grpstats(T, iG, {@mean});
        se = grpstats(T, iG,{@(x) nanstd(bootstrp(200,@(x) mean(x),x))});

        G = {a};
        [p,t] = anovan(T',G,'display','off');
    end

    % plot
    fig2 = figure('name',[ctype ' tuning by area']);
    xx = 1:numel(uarea);
    
    if 0
        hb = barwitherr(se,xx,mu);
        set(hb,'facecolor',ones(1,3)*0.8)
    else
        errorbar(xx,mu,se)
    end
    set(gca,'xtick',xx,'xticklabel',uarea)
    xtickangle(30)
    axis square

    str = sprintf('%s tuning (anova F) of single units per area\n ANOVA p=%.3g,F=%.3g',...
        ctype,t{2,7},t{2,6});
    title(str)
    ylabel('mean T')


    % output
    out = [];
    out.ctype = ctype;
    out.MU = MU;
    out.MUr = MUr;
    out.P = P;
    out.R = R;
    out.F = F;
    out.Pr = Pr;
    out.Fr = Fr;
    out.udat = udat;

    hout = [];
    hout.fig_tuning = fig1;
    hout.fig_area = fig2;
end









%% MISC
