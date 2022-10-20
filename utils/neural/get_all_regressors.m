function BAD = get_all_regressors(monk,datadir)
% BAD = get_all_regressors(monk,datadir)

%monk = datasets(1).name(1:2);
%[datadir,~] = fileparts(anadir);

nparallel = 20;

if nparallel>1 % parallel
    if isempty(gcp('nocreate'))
        myPool = parpool(nparallel);
    end
    
    spmd
       BAD_tmp = wrap_func(datadir,monk);
    end
    BAD = cat(1,BAD_tmp{:});
    
    foo=1;
else %serial
    BAD = wrap_func(datadir,monk);
end


sname = [datadir '/BAD_reg_build_' monk '.mat'];
save(sname,'BAD')


%% wrapper
function BAD=wrap_func(datadir,monk)

%paths
%posedir = [datadir '/Data_proc_13joint/data'];
posedir = [datadir '/Data_proc_13joint/data_ground'];

% settings
%fs_new = 1000;
fs_new = 30;

% get datasets
if strcmp(monk,'yo')
    %s = [get_code_path() '/bhv_cluster/data_log_yoda_ephys_bv.xlsx'];
    s = [get_code_path() '/bhv_cluster/data_log_yoda_ephys.xlsx'];
else
    s = [get_code_path() '/bhv_cluster/data_log_woodstock_ephys.xlsx'];
end
taskInfo = readtable(s);
d = taskInfo.name;
%d=d(10);

% loop over datasets
workerIndices = splitjobs('none',1:numel(d),numlabs);
IDX = workerIndices{labindex};

BAD = {};
for id=IDX
    %name = d(id).name;
    name = d{id};
    fprintf('%g: %s\n',id,name)
    
    % get evt regressors
    try
        out = build_evt_regressors([datadir '/' name],fs_new);
        t = out.info.time;
        
        % load preprocessed pose
        name2 = [posedir '/' name '_proc.mat'];
        load(name2)

        % add missing samples
        N = max(data_proc.frame);
        frameIdx = data_proc.frame+1;
        frame = [0:data_proc.frame(end)];
        t_pose = frame ./ 30;% * fs_new;

        % figure out where data is missing, in the new time
        isbad = ones(size(t_pose));
        isbad(frameIdx) = 0;
        isbad2 = interp1(t_pose,isbad,t,'linear','extrap');
        isbad2 = isbad2 > 0.5;
        
        % ------------------------------------------------
        % COM
        com = nan(numel(frame),3);
        com(frameIdx,:) = data_proc.com;
        
        % match com to evts
        icom = nan(numel(t),3);
        for ic=1:3
            icom(:,ic) = interp1(t_pose,com(:,ic),t,'linear','extrap');
        end
        icom(isbad2==1,:) = nan;

        % normalize range
        icom = (icom - min(icom)) ./ (max(icom) - min(icom));
        
        % diagnostic
        if 0
        figure
        set_bigfig(gcf,[0.2 0.2])
        plot(t_pose,com(:,1))
        hold all
        plot(t,icom(:,1))
        legend({'orig com','new com'})
        end

        % append
        out.com_x = icom(:,1);
        out.com_y = icom(:,2);
        out.com_z = icom(:,3);

        % ------------------------------------------------
        % joints
        
        ds = nan(numel(frame),size(data_proc.data,2));
        ds(frameIdx,:) = data_proc.data;
        
        % match to evts
        ds2 = nan(numel(t),3);
        for ic=1:size(ds,2)
            ds2(:,ic) = interp1(t_pose,ds(:,ic),t,'linear','extrap');
        end
        ds2(isbad2,:) = nan;
        
        % normalize
        ds2 = (ds2 - min(ds2)) ./ (max(ds2) - min(ds2));
        
        % append
        labs = cellfun(@(x) ['jt_' x],data_proc.labels,'un',0);
        for ij=1:numel(labs)
            out.(labs{ij}) = ds2(:,ij);
        end
        
        % remove nans
        
        % ------------------------------------------------
        % save
        sname = [datadir '/' name '/regressors.mat'];
        save(sname,'-struct','out')

        foo=1;
    catch err
        warning('fail')
        ib = size(BAD,1)+1;
        BAD{ib,1} = name;
        BAD{ib,2} = err;
        BAD{ib,3} = id;
    end
end

