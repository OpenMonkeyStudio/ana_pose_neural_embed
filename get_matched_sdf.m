function get_matched_sdf(anadir,monk)
% get_matched_sdf(anadir,monk)

% settings
nparallel = 20;

fs_spk = 1000;
fs_frame = 30;

collapseToMUA = 0; % 0=dont, 1=gross subvidions, 2=minor sibdivisions
regressOutVariables = 1;
    theseRegressors = {'evt','com'}; % 'jt'
resid_type = 2; %1=linear, 2=poisson
timwin = [-0.1 0.1];
timwin_regressor = [-0.025 0.025];
    
% paths
posedir = [fileparts(anadir) '/Data_proc_13joint/data_ground'];
[spkparentpath,~] = fileparts(anadir); % one level up

tmpstr = {'lin','pois'};
muastr = {'','_mua1','_mua2'};
sdfpath = sprintf('%s/sdf_fs30_resid_%s_%g-%g%s',anadir,tmpstr{resid_type},timwin(1)*1000, timwin(2)*1000,muastr{collapseToMUA+1});
if ~exist(sdfpath); mkdir(sdfpath); end
sdfpath_data = [sdfpath '/sdf'];
if ~exist(sdfpath_data); mkdir(sdfpath_data); end

% datasets
[datasets,taskInfo] = get_datasets(monk);


%% get SDF for each cell    
%ignore_warnings = {'MATLAB:singularMatrix','stats:LinearModel:RankDefDesignMat','FieldTrip:ft_spikedensity:ft_checkconfig'};
ignore_warnings = {'all'};
for iw=1:numel(ignore_warnings); warning('off',ignore_warnings{iw}); end

% start parallel pool
if nparallel > 1 && isempty(gcp('nocreate'))
    myPool = parpool('local',nparallel);
end

% call
startTime = tic;
parfor id=1:numel(datasets)
%for id=1:numel(datasets)
    name = datasets(id).name;
    fprintf('%g: %s\n',id,name)

    try
        % figure out the frames to match
        fprintf('loading pose data...\n')
        tmp = load([posedir '/' name '_proc.mat']);
        thisFrame = tmp.data_proc.frame;

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

        
        % now, loop through each cell and get SDF
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

            cfg = [];
            cfg.timwin = timwin;
            cfg.latency = tlim;
            cfg.fsample = fs_frame;
            sdf = ft_spikedensity(cfg,spk);


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
                fprintf('\t getting area...\n')
                [a,~] = get_area(ch,day,monk,0);
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
                tmp.id = id;
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

% finish and clean
fprintf('\n\n TOTAL SDF calc: %g\n', toc(startTime))     

for iw=1:numel(ignore_warnings); warning('on',ignore_warnings{iw}); end
delete(gcp('nocreate'))
    