function [SDF,sdfnames] = load_back_sdf(sdfpath,useSDF_resid,theseDays,theseChan,theseAreas,useGrossSubdivisions,forceAreaUpdate)
% [SDF,sdfnames] = load_back_sdf(sdfpath,useSDF_resid,theseDays,theseChan,theseAreas,useGrossSubdivisions,forceAreaUpdate)

tic

% checks
if nargin < 3
    theseDays = {};
end
if nargin < 4
    theseChan = [];
end
if nargin < 5
    theseAreas = {};
    useGrossSubdivisions = 0;
end
if nargin < 6
    useGrossSubdivisions = 1;
end
if nargin < 7
    forceAreaUpdate = 1;
end

% get datasets
d = dir([sdfpath '/*_sdf_matched.mat']);
d = natsortfiles({d.name});

if ~isempty(theseDays)
    theseDays = cellfun(@(x) datestr(x,'yyyy-mm-dd'),theseDays,'un',0);
    seld = contains(d,theseDays);
    d(~seld) = [];
end

if ~isempty(theseChan)
    tmpchan = cellfun(@(x) sprintf('_ch%g_',x),num2cell(theseChan),'un',0);
    seld = contains(d,tmpchan);
    d(~seld) = [];
end

% loop over days
fprintf('loading sdf files back')

SDF=[];
sdfnames = {};
ilast = 0;
for id=1:numel(d)
    dotdotdot(id,0.1,numel(d))
    name = d{id};
    name_mdl= [name(1:end-16) '_regmdl.mat'];
        
    % load
    tmp = load([sdfpath '/' name]);
    tmpmdl = load([sdfpath '/' name_mdl]);
    
    % selection
    if tmpmdl.Rsquared.Deviance < 0
        %warning('bad fit')
        continue
    end
    if strcmp(tmp.area,'?')
        %warning('bad area')
        continue
    end
    
    % mean sdf
    s = tmp.sdf;
    mu = nanmean(s);

    % original or residualized sdf?
    if useSDF_resid
        tmp.sdf = tmp.sdf_resid;
    end
    try, tmp = rmfield(tmp,'sdf_resid'); end

    tmp.mean_rate = mu;
    tmp.name = name;

    % append
    ilast = ilast+1;
    if numel(SDF)==0 % init at full size
        SDF = repmat(tmp,numel(d),1);
        sdfnames{1} = name;
    else
        SDF(ilast) = tmp;
        sdfnames{ilast} = name;
    end
end

% cull last
SDF(ilast+1:end) = [];

% did anything save?
if numel(SDF)==0
    return
end

% update areas, just in case
if forceAreaUpdate
    fprintf('\t updating areas...\n')
    ch = [SDF.ch];
    days = {SDF.day};
    monk = SDF(1).name(1:2);
    [areas,noRecord] = get_area(ch,days,monk,1);
    for ii=1:numel(SDF)
        SDF(ii).area = areas(ii);
    end
else
    areas = [SDF.area];
end

% cull areas we dont need (check for gross and sub- divisions)
if numel(theseAreas)>0
    fprintf('\t culling areas...\n')
    if useGrossSubdivisions
        areas2 = collapse_areas(areas);
        del = ~contains(lower(areas2),lower(theseAreas));
    else
        del = ~contains(lower(areas),lower(theseAreas));
    end
    SDF(del) = [];
    sdfnames(del) = [];
end

% final check
if numel(SDF)==0
    error('no SDF left after culling!')
end

toc
end