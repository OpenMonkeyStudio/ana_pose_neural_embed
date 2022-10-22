anadir1 = '/mnt/scratch/BV_embed/P_neural_embed/embed_rhesus';
anadir2 = '/mnt/scratch/BV_embed/P_neural_final/embed_rhesus_yo';

if 0
    name = 'yo_2021-02-05_01_enviro';
    dat1 = load([anadir1 '/X_feat_norm/' name '_proc_feat.mat']);
    dat2 = load([anadir2 '/X_feat_norm/' name '_feat.mat']);
    
    umap1 = load([anadir1 '/umap_train.mat']);
    umap2 = load([anadir2 '/umap_train.mat']);
end

x1 = umap1.embedding_;
x2 = umap2.embedding_;

dx = x1-x2;
d = sqrt(sum(d.^2,2));

