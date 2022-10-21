function [Y,procInfo] = call_umap(X,cfg,procInfo)
% [Y,procInfo] = embed_pose2(X,cfg)
% [Y,procInfo] = embed_pose2(X,cfg,procInfo)

% checks

cfg = checkfield(cfg,'anadir','needit'); % where the daata will save
cfg = checkfield(cfg,'savefig',1);
cfg = checkfield(cfg,'train','needit'); % is this a training or test data
cfg = checkfield(cfg,'nparallel',0); % is this a training or test data
cfg = checkfield(cfg,'presave',1); % is this a training or test data
cfg = checkfield(cfg,'group',[]); % if multiple datasets are concatnated
cfg = checkfield(cfg,'pca_indiv',1);
cfg = checkfield(cfg,'pca_global',1);
    cfg = checkfield(cfg,'pca_global_pcdims',[]);
    cfg = checkfield(cfg,'pca_global_thresh',[]);
    cfg = checkfield(cfg,'pca_global_type','cov');
    cfg = checkfield(cfg,'pca_global_rotatefactors','');
cfg = checkfield(cfg,'indiv_factorn',0);
cfg = checkfield(cfg,'group_ica',numel(unique(cfg.group))>1);
cfg = checkfield(cfg,'wvt',1);
cfg = checkfield(cfg,'wvt_pca',0);
cfg = checkfield(cfg,'umap',1);
    cfg = checkfield(cfg,'umap_class','umap');
    cfg = checkfield(cfg,'umap_cfg',[]);
cfg = checkfield(cfg,'cluster',1);
    cfg = checkfield(cfg,'cluster_method','WATERSHED');

if cfg.wvt || cfg.wvt_pca
    cfg = checkfield(cfg,'fs','needit');
end


% settings
thresh = 95; % 95

loadInput = iscell(X);

% extract
anadir = cfg.anadir;
igrp = cfg.group;
if isempty(igrp)
    igrp = ones(size(X,1),1);
end

[grp,~,igrp] = unique(igrp);

% define paths
[~,~,pyenvpath] = set_pose_paths(0);

figdir = [anadir '/Figures'];
if ~exist(anadir); mkdir(anadir); end
if ~exist(figdir); mkdir(figdir); end

if cfg.train
    sname_procInfo = [anadir '/procInfo_train.mat'];
else
    sname_procInfo = [anadir '/procInfo_test.mat'];
end

% start parallel pool?
if cfg.nparallel > 1 && isempty(gcp('nocreate'))
    parpool(cfg.nparallel)
end

% init
cd(anadir)

Y = X;
if ~loadInput
    [nsmp,nfeat] = size(Y);
    fprintf('\ninput is %g x %g\nsaving to: %s\n\n',nsmp,nfeat,cfg.anadir)
else
    fprintf('\nloading presaved input... \n\n')
end

% save input
procInfo.grp = grp;
procInfo.igrp = igrp;
procInfo.cfg = cfg;
save(sname_procInfo,'-struct','procInfo')

if cfg.presave==1
    fprintf('presaving training data... \n')
    
    if cfg.train
        save([anadir '/X_train_in.mat'],'X','-v7.3')
    else
        save([anadir '/X_test_in.mat'],'X','-v7.3')
    end
end


%% save before embedding
if 1 && ~loadInput
    fprintf('saving pre-embedding data..\n')
    if cfg.train
        sname = [anadir '/Y_train.mat'];
        save(sname,'Y','-v7.3')
    else
        sname = [anadir '/Y_test.mat'];
        save(sname,'Y','-v7.3')
    end
end

%% call embeddiing
if cfg.umap
    verbose = 1;
    class = lower(cfg.umap_class);
    umap_cfg = cfg.umap_cfg;
    
    if cfg.train
        tic
        disp('embedding train...')
    
        % prepare data
        %umap_cfg = checkfield(umap_cfg,'class','umap');
        umap_cfg.X_train = Y;
        
        % supervised?
        if isfield(umap_cfg,'target') && all(umap_cfg.target==-1)
            umap_cfg = rmfield(umap_cfg,'target');      
        end

        % python inputs
        cmd = 'fit';
        name_in = [anadir '/umap_data_train.mat'];
        savepath = anadir;

        % send to python
        fprintf('\t presaving for transfer \n')
        %save(name_in,'-struct','umap_cfg')
        save(name_in,'-struct','umap_cfg','-v7.3')

        if strcmp(class,'umap')
            func = [get_code_path() '/bhv_cluster/matlab/python/call_umap.py'];
        elseif strcmp(class,'alignedumap')
            func = [get_code_path() '/bhv_cluster/matlab/python/call_umap_aligned.py'];
        else
            error('%s not recognized',class)
        end

        commandStr = sprintf('%s %s %s %s %s',pyenvpath,func,cmd,name_in,savepath);
        fprintf('\t executing: %s \n',commandStr)

        if verbose
            [status,result] = system(commandStr,'-echo');
        else
            [status,result] = system(commandStr);
        end

        % wass it succesful?
        if status~=0 % fail
            error('FAIL')
        end
        toc
        
        % load back
        if strcmp(class,'umap')
            umap_train = load([savepath '/umap_train.mat']);
            Y = umap_train.embedding_;        

            % udpate
            delThis = {'embedding_','graph_','graph_dists_',};
            for ii=1:numel(delThis)
                try
                    umap_train = rmfield(umap_train,delThis{ii});
                catch
                    fprintf('no umap field: %s\n',delThis{ii})
                end
            end
            procInfo.umap.train.umap = umap_train;
            procInfo.umap.train.savepath = savepath;
            procInfo.umap.train.name_in = name_in;
        elseif strcmp(class,'alignedumap')
            umap_train = load([savepath '/umap_train.mat']);
            Y = umap_train.embeddings_;  
            
            procInfo.umap.train.savepath = savepath;
            procInfo.umap.train.name_in = name_in;
        end
        save(sname_procInfo,'-struct','procInfo')

    else
        fprintf('re-embedding test data, using %s...\n',umap_cfg.knntype)
        tic
        K = 20;
        
        % load the training data
        Y_train = load([anadir '/Y_train.mat']);
        Y_train = Y_train.Y;
        
        umap_train = load([anadir '/umap_train.mat']);
        metric = umap_cfg.metric;
        Yu_train = umap_train.embedding_;

        % re-embed using KNN search
        if strcmp(umap_cfg.knntype,'faiss')
            % prep batches
            if ~loadInput
                yq = cellfun(@(x) Y(igrp==x,:),num2cell(unique(igrp)),'un',0);
                [idx,D,~] = call_faiss_knn(Y_train,{yq,[anadir '/knn_fais_batch']},K,metric,umap_cfg.useGPU);
            else
                [idx,D,~] = call_faiss_knn(Y_train,{Y,{}},K,metric,umap_cfg.useGPU);
            end
        else % default to matlab
            fprintf('\t matlab single core ...\n')
            [idx,D] = knnsearch(Y_train,Y,'K',K,'distance',metric);
        end
        
        tmp_test = Yu_train(idx,:);
        tmp_test = reshape(tmp_test,[numel(cfg.group), K, size(Yu_train,2)]);
        Y_test = squeeze(nanmedian(tmp_test,2));
    
        % save
        fprintf('\t saving cluster umap...\n')

        umap_test = [];
        umap_test.embedding_ = Y_test;
        umap_test.K = K;
        umap_test.type = 'knnsearch';
        umap_test.metric = metric;
        umap_test.idx_train = idx;
        umap_test.D_train = D;
        sname = [anadir '/umap_test.mat'];
        save(sname,'-struct','umap_test','-v7.3')
    
        % finish
        Y = Y_test;
        foo=1;
        
        toc
        %{
        tic
        disp('embedding test...')
    
        % prepare data
        umap_cfg = cfg.umap_cfg;
        umap_cfg.X_test = Y;

        
        % python inputs
        cmd = 'transform';
        name_in = [anadir '/umap_data_test.mat'];
        savepath = anadir;

        % send to python
        save(name_in,'-struct','umap_cfg')

        if strcmp(class,'umap')
            func = [get_code_path() '/bhv_cluster/matlab/python/call_umap.py'];
        elseif strcmp(class,'alignedumap')
            func = [get_code_path() '/bhv_cluster/matlab/python/call_umap_aligned.py'];
        else
            error('%s not recognized',class)
        end

        commandStr = sprintf('%s %s %s %s %s',pyenvpath,func,cmd,name_in,savepath);

        if verbose
            [status,result] = system(commandStr,'-echo');
        else
            [status,result] = system(commandStr);
        end

        % wass it succesful?
        if status~=0 % fail
            error('FAIL')
        end
        toc
        
        % load back
        if strcmp(class,'umap')
            umap_train = load([savepath '/umap_train.mat']);
            Y = umap_train.embedding_;        

            % clean
            ignore = {'embedding_','graph_','graph_dists_'};
            for ii=1:numel(ignore)
                try
                    umap_train = rmfield(umap_train,ignopre{ii});
                catch
                    warning('%s field doesnt exist un umap object',ignore{ii})
                end
            end
            
            % udpate
            procInfo.umap.test.umap = umap_train;
            procInfo.umap.test.savepath = savepath;
            procInfo.umap.test.name_in = name_in;
        elseif strcmp(class,'alignedumap')
            umap_train = load([savepath '/umap_train.mat']);
            Y = umap_train.embeddings_;  
            
            procInfo.umap.test.savepath = savepath;
            procInfo.umap.test.name_in = name_in;
        end
        save(sname_procInfo,'-struct','procInfo')
    %}
    end
end


%% cluster
if cfg.cluster
    if cfg.train
        fprintf('clustering via %s\n',cfg.cluster_method)
        
        tic
        [clabels, Ld, Lbnds,outClust] = examine_clusters(Y, cfg.cluster_method);
        toc
        
        sname = [anadir '/cluster_train.mat'];
        save(sname,'clabels','Ld', 'Lbnds','outClust')
    else
        fprintf('clustering test data using KNN...\n')
        
        tic
        % test data
        cluster_train = load([anadir '/cluster_train.mat']);
        umap_test = load([anadir '/umap_test.mat']);
        umap_train = load([anadir '/umap_train.mat']);
        Y = umap_test.embedding_;
        Y_train = umap_train.embedding_;
        
        % find the state labels
        fprintf('\t finding state labels...\n')
        xv = cluster_train.outClust.xv;
        yv = cluster_train.outClust.yv;

        if 0
            clabels_test = double(interp2(xv,yv,cluster_train.Ld,Y(:,1),Y(:,2),'nearest'));
        else
            IDX = knnsearch(Y_train,Y,'K',10);
            clabels_test = cluster_train.clabels(IDX);
            clabels_test = mode(clabels_test,2);
        end

        % get heatmap again
        fprintf('\t getting density estimate...\n')
        [tmp1,tmp2] = meshgrid(xv,yv);
        [dens, ~, bw] = ksdens(Y,[tmp1(:) tmp2(:)]);
       
        fprintf('\t saving cluster info...\n')
        cluster_test = cluster_train;
        cluster_test.clabels = clabels_test;
        cluster_test.outClust.dens2 = dens;

        sname = [anadir '/cluster_test.mat'];
        save(sname,'-struct','cluster_test')

        toc
    end
end

%% final
save(sname_procInfo,'-struct','procInfo')