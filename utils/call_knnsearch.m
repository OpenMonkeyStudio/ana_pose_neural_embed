function [Y_test,cfg] = call_knnsearch(X_train,Y_train,X_test,cfg)

% checks
cfg = checkfield(cfg,'knntype','faiss');
    cfg = checkfield(cfg,'useGPU',1);
cfg = checkfield(cfg,'K',20);

% are we going to do it in batches?
doBatches = iscell(X_test);

fprintf('re-embedding test data, using %s...\n',cfg.knntype)
tic

% load the data

Yu_train = umap_train.embedding_;

% re-embed using KNN search
if strcmp(cfg.knntype,'faiss')
    % prep batches
    if ~doBatches
        yq = cellfun(@(x) Y(igrp==x,:),num2cell(unique(igrp)),'un',0);
        [idx,D,~] = call_faiss_knn(X_train,{yq,[anadir '/knn_fais_batch']},cfg.K,cfg.metric,cfg.useGPU);
    else
        [idx,D,~] = call_faiss_knn(X_train,{X_test,{}},K,metric,cfg.useGPU);
    end
else % default to matlab
    fprintf('\t matlab single core ...\n')
    [idx,D] = knnsearch(X_train,X_test,'K',cfg.K,'distance',cfg.metric);
end
        
tmp_test = Y_train(idx,:);
tmp_test = reshape(tmp_test,[numel(cfg.group), cfg.K, size(Y_train,2)]);
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
    end
end
