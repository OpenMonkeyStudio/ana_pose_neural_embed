function call_cluster_hier(PO,mpath,dataname)
    [~,~,pyenvpath,~] = set_pose_paths(0);

    % call python
    verbose = 1;
    cmd = 'fit_paris';
    savepath = mpath;

    % send to python
    name_in = [dataname '_in.mat'];

    dat = [];
    dat.po = PO;
    save(name_in,'-struct','dat')
    %save(name_in,'-struct','dat','-v7.3')

    func = [get_code_path() '/bhv_cluster/matlab/python/call_transition_cluster.py'];
    commandStr = sprintf('%s %s %s %s %s',pyenvpath,func,cmd,dataname,savepath);

    tic
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

end
