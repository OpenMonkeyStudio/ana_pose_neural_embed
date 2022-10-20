monk = 'wo';

datadir = '/mnt/scratch3/BV_embed/P_neural_embed_wo/Data_proc_13joint/data_ground';
cd(datadir)

%% loop through each dataset to get data
if 0
    dd = dir([datadir '/' monk '*']);
    dat = [];
    
    for id=1:numel(dd)
        % load
        name = dd(id).name;
        fprintf('%g: %s \n',id,name)
        load(name)
    
        % extract
        com = data_proc.com;
        t = data_proc.frame'/30;
    
        v = sum(diff(com).^2,2) ./ diff(t);
        v = [0; v];
    
        h = com(:,2);
    
        % store
        tmp = [];
        tmp.v = v;
        tmp.com = com;
        tmp.h = h;
    
        dat = cat(1,dat,tmp);
    
        foo=1;
    end
end

%% determine stats on each dataset to find good candidates
hThresh = 3;

P = [];
for id=1:numel(dat)
    n = size(dat(id).com,1);
    p = dat(id).h > hThresh;
    P(id) = sum(p)./n;
end