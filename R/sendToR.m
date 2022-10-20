function out = sendToPython(dat,opts)
% out = sendToR(dat,opts)
% 
% Calls R unction from matlab. "dat" has all data and inputs, that
% will be parsed by the python function. "dat" is saved in a temporary location. 
% The python func must include two inputs, the input name where the data is
% saved, and and output name where results will be resaved.
%
% dat: includes all data and inputs to the python func. Python func then
% has to parse it
%
% funcpath: fullpath to the python function to call eg
% /Users/<name>/myFunc.py
%
% innspired by code from Runyan et al 2017

% checks
opts = checkfield(opts,'verbose',0);
opts = checkfield(opts,'clean_files',0);
opts = checkfield(opts,'tmpname','');
opts = checkfield(opts,'func','needit');
opts = checkfield(opts,'envpath','needit');
opts = checkfield(opts,'add_orig_data',0);

verbose = opts.verbose;

func = opts.func;
if ~strcmp( func(end-2:end), '.R')
    func = [func '.R'];
end
[funcpath,func] = fileparts(func);

envpath = opts.envpath;

fprintf('***  calling R  ***\n%s\n',func);

% prepare file names
if ~isempty(opts.tmpname)
    tmpname = opts.tmpname;
else
    tmpname = tempname;
end
name_in = [tmpname '_in.mat'];
name_out = [tmpname '_out.mat'];

%save it
save(name_in,'-struct','dat')

% build command string 
cmd = sprintf('cd %s; %s %s "%s"',funcpath,envpath,func,tmpname);

if verbose
    [status,result] = system(cmd,'-echo');
else
    [status,result] = system(cmd);
end

% reload output
if status==0
    %load back the data
    out = load(name_out);

    % add the original data
    if opts.add_orig_data
        f = fieldnames(dat);
        for ii = 1:length(f)
            if ~isfield(out,f{ii})
                out.(f{ii}) = dat.(f{ii});
            else
                warning('field "%s" already exists, renaming',f{ii})
                newf = [f{ii} '_new'];
                out.(newf) = dat.f{ii};
            end
        end
    end
    
    if opts.clean_files
        doClean(name_in,name_out)
    end
else
    warning('python call failed')
    if ~verbose
        disp(result)
    end
    
    out = [];
    if opts.clean_files
        doClean(name_in,name_out)
    end
end

foo=1;

%% MISC
function doClean(name_in,name_out)
try, delete(name_in), end
try, delete(name_out), end