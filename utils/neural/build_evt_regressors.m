function out = build_evt_regressors(datapath,fs_new)
% out = build_evt_regressors(datapath,fs_new)

%% settings
%fs_new = 1;
theseEvents = {'timeoutOn','rewOn','cueOn','leverPress','screenOn'};
%theseEvents = {'timeoutOn','rewOn','cueOn','leverPress'};
%theseEvents = {'screenOn'};


%% laod evt, info
cd(datapath)
load('info.mat')
load('evt.mat')
if 0
    d = dir('*Enviro_feeder.mat');
    load(d(1).name)
else
    patchInfo.id = evt.info.feeder;
    patchInfo.ecu_line = evt.info.ecu_lines;
end

%% stuff
ecuLine = [patchInfo.ecu_line];
thesePatches = [patchInfo.id];

fs = info.fs;
firstLast = info.firstLast;
firstLastFrame = info.firstLastFrame;

if 0 % only during capture
    st = (firstLastFrame(1) - firstLast(1))./fs * fs_new;
    fn = (firstLastFrame(2) - firstLast(2))./fs * fs_new;
    fn = (firstLast(2) - firstLast(1))./fs * fs_new +fn;
    thisTime = (0+st):1/fs_new:fn; 
else
    fn = (firstLast(2) - firstLast(1))./fs;
    thisTime = 0:1/fs_new:fn; 
end

% init
tmp = [];
tmp.time = thisTime;
tmp.info = info;
tmp.fs = fs_new;
tmp.ecu_line = ecuLine;
tmp.patchID = thesePatches;
tmp.datapath = datapath;
tmp.empty_evt = isempty(evt);
if tmp.empty_evt
    tmp.missing_evts = 1;
else
    a = ~ismember(thesePatches,unique(evt.patchID));
    tmp.missing_evts = any(a);
    tmp.ecu_lines_missing_evts = a;
    foo=1;
end

out = [];
out.info = tmp;

%% build each regressor, for each patch
for it = 1:numel(theseEvents)
    s = theseEvents{it};
    
    for ip=1:numel(thesePatches)
        r = zeros(numel(thisTime),1);
        ecu = ecuLine(ip);

        if isempty(evt)
            warning('empty evt file!')
        else
            selpatch = evt.patchID==thesePatches(ip);
            t = evt.timestamp;

            % define start and end of each event
            if strcmp(s,'timeoutOn')
                st = t( evt.timeoutStart & selpatch );
                fn = t( evt.timeoutEnd & selpatch );
            elseif strcmp(s,'rewOn')
                st = t( evt.rewStart & selpatch );
                fn = t( evt.rewEnd & selpatch );
            elseif strcmp(s,'cueOn')
                st = t( evt.cueOn & selpatch );
                fn = t( evt.cueOff &selpatch );
            elseif strcmp(s,'leverPress')
                fn = t( (evt.leverPress & selpatch) ); % end of press
                st = fn - 0.2*fs; % start of press
            elseif strcmp(s,'screenOn')
                st = t( evt.screenOn & selpatch );
                fn = t( evt.screenOff & selpatch );

                foo=1;
            else
                error('unrecognized event: %s',s)
            end

            % checks
            if ~(isempty(st) || isempty(fn))
                if fn(1) < st(1); fn(1) = []; end
                if numel(st) > numel(fn)
                    fn(end+1) = firstLast(end);
                end
                if numel(st) ~= numel(fn)
                    error('not matching')
                end

                % build out regrssors
                st2 = (st-firstLast(1)) ./ fs;
                fn2 = (fn-firstLast(1)) ./ fs;

                for is=1:numel(st2)
                    sel = thisTime >= st2(is) & thisTime <= fn2(is);
                    r(sel) = 1;
                end
            else
                warning('no %s events on ECU line %g',s,ecu)
            end
        end

        % store
        s2 = sprintf('evt_%s_ecu%g',s,ecu);
        out.(s2) = r;
    end
end

    
%% save
%sname = [datapath '/evt_time.mat'];
%save(sname,'-struct','out')

foo=1;