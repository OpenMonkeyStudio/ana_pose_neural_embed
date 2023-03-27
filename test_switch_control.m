theseSettings = 2:4;

SEG_all = {};
for im = 1:2
    
    % load
    monk = monks{im};
    anadir = anadirs{im};

    load_pose_neural_data 
        
    for ia=theseSettings
        figdir = [anadirs{im} '/Figures_' monk '_switch' num2str(ia)];
        if ~exist(figdir); mkdir(figdir); end

        % ------------------------------------------------------------
        % action switch encoding
        if 1
            % switch sdf (Figure 6B)
            cfg = [];
            cfg.sdfpath = sdfpath;
            cfg.figdir = figdir; %[figdir '_controlSwitch'];
            cfg.datasets = datasets;
            cfg.fs_frame = fs_frame;
            cfg.uarea = uarea;
            cfg.plot = 1;
            cfg.only_nonengage = 0;
            
            if ia==1
                cfg.plot_lim = [-1 1];
                cfg.seg_lim = [-1 1];
                cfg.seg_min = 0.2; % 0.2
            elseif ia==2
                cfg.plot_lim = [-5 20];
                cfg.seg_lim = [-1 1];
                cfg.seg_min = 0.2; % 0.2
            elseif ia==3
                cfg.plot_lim = [-5 20];
                cfg.seg_lim = [-1 1];
                cfg.seg_min = 0.5; % 0.2
            elseif ia==4
                cfg.plot_lim = [-5 20];
                cfg.seg_lim = [-1 1];
                cfg.seg_min = 1; % 0.2
            end
            cfg.eng_smooth = 1;

            cfg.avgtype = 'median';
            cfg.normtype = 'preswitch';
            cfg.weighted_mean = 0;

            out_switch = ana_switch_sdf(cfg,SDF,C,frame,iarea,idat);
            set(gcf,'name',sprintf('%s: lims=%g',monk,ia))
            
            % switch sdf + controls (Figure 6C)
            %cfg.only_nonengage = 1;
            %cfg.weighted_mean = 1;
            %out_switch2 = ana_switch_sdf(cfg,SDF,C,frame,iarea,idat);


            % store for later
            SEG_all{im,1,ia} = out_switch;
            %SEG_all{im,2,ia} = out_switch2;
        end
    end
end

%% plot overall SDF
for ia=theseSettings
    doWeightedMean = 1;
    
    figdir2 = [anadirs{1} '/Figures_bothMonk_switch' num2str(ia) '_weight' num2str(doWeightedMean)];
    if ~exist(figdir2); mkdir(figdir2); end

    % plot
    iseg = 1;
    res_tmp = [SEG_all{1,iseg,ia}.RES_seg; SEG_all{2,iseg,ia}.RES_seg];

    cfg = [];
    cfg.figdir = figdir2;
    cfg.fs_frame = fs_frame;
    cfg.uarea = uarea;
    cfg.plot_lim = SEG_all{1,iseg,ia}.cfg.plot_lim;
    cfg.seg_lim = SEG_all{1,iseg,ia}.cfg.seg_lim;
    
    cfg.avgtype = 'median';
    cfg.normtype = 'preswitch';
    cfg.weighted_mean = doWeightedMean;

    tmp = ana_switch_sdf(cfg,res_tmp);
end