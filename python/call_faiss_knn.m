function [IDX,D,out] = call_faiss_knn(X,Xq,K,metric,useGPU)
% [IDX,D,out] = call_faiss_knn(X,Xq,K,metric,useGPU)
% [IDX,D,out] = call_faiss_knn(X,{{Xq1...Xqn},batchsavepath},K,metric,useGPU)

[~,~,pyenvpath,~,~,codepath] = set_pose_paths(0);

% prep query points
if iscell(Xq)
    batchpath = Xq{2};
    
    if ~isempty(batchpath) % have to create batches
        Xq = Xq{1};
        if ~exist(batchpath); mkdir(batchpath); end

        fprintf('preparing batches: ')
        Xq2 = {};
        for id=1:numel(Xq)
            fprintf('%g,',id)
            sname = sprintf('%s/batch%g.mat',batchpath,id);
            tmp = [];
            tmp.Xq = Xq{id};
            parsave(sname,tmp)
            Xq2{id} = sname;
        end
    else
        Xq2 = Xq{1};
    end
    fprintf('\n')
else
    Xq2 = Xq;
end
clear Xq

% call faiss
fprintf('calling faiss KNN...\n')
dat = [];
dat.X = X;
dat.Xq = Xq2;
dat.K = K;
dat.metric = metric;
dat.useGPU = useGPU;

opts = [];
opts.func = [codepath '/bhv_cluster/matlab/python/call_knnsearch.py'];
opts.verbose = 0;
opts.clean_files = 1;
opts.pyenvpath = pyenvpath;

out = sendToPython(dat,opts);

% final output
% - make sure its in the proper format
if ~iscell(out.distances)
    sz = size(out.neighbors);
    
    IDX = reshape( permute(out.neighbors,[2 1 3]), [sz(1)*sz(2) sz(3)] ) + 1;
    D = reshape( permute(out.distances,[2 1 3]), [sz(1)*sz(2) sz(3)] );
else
    IDX = cat(1,out.neighbors{:})+1;
    D = cat(1,out.distances{:});
end


