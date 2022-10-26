function [labels, Ld, Lbnds,out] = examine_clusters(data, clst_method)

	%% This function includes the types of clustering that I have tested. 
    out = [];
    out.method = clst_method;

	switch clst_method
		case 'HDBSCAN'
			clusterer = HDBSCAN(data);
			%  clusterer.minpts	= 15; clusterer.minclustsize = 15; clusterer.outlierThresh = 0.95; clusterer.minClustNum = 1;
			clusterer.fit_model(); 			% trains a cluster hierarchy

			clusterer.get_best_clusters(); 	% finds the optimal "flat" clustering scheme
			clusterer.get_membership();		% assigns cluster labels to the points in X

			% we can also visualize the tree and/or actual clusters in a 2D or 3D space (depending on self.nDims)
			clusterer.plot_tree();
			clusterer.plot_clusters(); % plots a scatter of the points, color coded by the associated labels

			labels	 = clusterer.labels;

            out.clusterer = clusterer;
		case 'AGGLO'
			Z	= linkage(data, 'ward');%'average','chebychev');
			labels	= cluster(Z, 'cutoff', 3, 'depth', 4); %,'maxclust', 6);

			cutoff = median([Z(end-2,3) Z(end-1,3)]);
			dendrogram(Z,'ColorThreshold',cutoff);
		case 'WATERSHED'
			%perc_rng = [10,95]; % Previous result was based on [15, 80]
			perc_rng = [1,99]; % Previous result was based on [15, 80]
			
            if 0 % scotts rule
                sig = mad(data,1,1) / 0.6745;
                [N, d] = size(data);
                bw = sig * (4/((d+2)*N))^(1/(d+4));
                bw = bw * 0.9;
            else
                bw = [];
            end
            pts = 200; %120
			[dens, pts, bw] = ksdens(data(:,1:2),pts,bw);
			xv	 = unique(pts(:,1)); yv = unique(pts(:,2));
			
			%dens2		= clip(dens, prctile(dens(:),perc_rng));
            dens2 = dens;
			[Ld, Lbnds] = watershed_seg(dens2);
			% Ld(dens < prctile(dens(:),25)) = 0;
			labels = double(interp2(xv,yv,Ld,data(:,1),data(:,2),'nearest'));
            
            out.xv = xv;
            out.yv = yv;
            out.dens2 = dens2;
            out.pts = pts;
            out.bw = bw;
	end
	
	if ~strcmp(clst_method, 'WATERSHED')
		Ld = []; 
		Lbnds = [];
	end
end