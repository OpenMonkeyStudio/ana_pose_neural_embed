%segMins = [0.2 0.5 1];
segMins = 0.2;

if 0

    SEG_all = {};
    for im = 1:2

        % load
        monk = monks{im};
        anadir = anadirs{im};

        load_pose_neural_data 

        for ia=1:numel(segMins)
            % switch sdf (Figure 6B)
            cfg = [];
            cfg.sdfpath = sdfpath;
            cfg.figdir = ''; %[figdir '_controlSwitch'];
            cfg.datasets = datasets;
            cfg.fs_frame = fs_frame;
            cfg.uarea = uarea;
            cfg.only_nonengage = 0;
            cfg.eng_smooth = 1;

            cfg.plot = 0;

            cfg.avgtype = 'median';
            %cfg.normtype = 'preavg'; 
            cfg.weighted_mean = 0;

            cfg.seg_lim = [-5 15];
            cfg.seg_min = segMins(ia); % 0.2


            out_switch = ana_switch_sdf(cfg,SDF,C,frame,iarea,idat);

            % store for later
            SEG_all{im,1,ia} = out_switch;
            %SEG_all{im,2,ia} = out_switch2;
        end
    end
    
end



%% plot overall SDF
for ia=1%numel(segMins)
    doWeightedMean = 0;
    
    if 0
        figdir2 = [anadirs{1} '/Figures_bothMonk_switch' num2str(ia) '_weight' num2str(doWeightedMean)];
        if ~exist(figdir2); mkdir(figdir2); end
    else
        figdir2 = '';
    end
    
    % plot
    iseg = 1;
    res_tmp = [SEG_all{1,iseg,ia}.RES_seg; SEG_all{2,iseg,ia}.RES_seg];

    cfg = [];
    cfg.figdir = figdir2;
    cfg.fs_frame = fs_frame;
    cfg.uarea = uarea;
    cfg.plot_lim = [-5 15];
    cfg.seg_lim = SEG_all{1,iseg,ia}.cfg.seg_lim;
    cfg.ana_lim = [-1 0; 0 1];
    cfg.baseline = [-5 15];
    
    cfg.avgtype = 'median';
    cfg.normtype = 'zindivseg'; % zindivseg, zavgseg, zsess,propindivseg, rngindivseg
    cfg.normavg = 'mean';
    cfg.weighted_mean = doWeightedMean;

    tmp = ana_switch_sdf(cfg,res_tmp);
end