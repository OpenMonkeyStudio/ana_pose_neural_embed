library(Seurat)
library(cowplot)

parpath='/Users/Ben/Documents/git/oms_internal/bhv_cluster/matlab/R/data'

library(Seurat)
library(cowplot)
ctrl.data <- read.table(file = paste(parpath,'immune_control_expression_matrix.txt.gz',sep='/'), sep = "\t")
stim.data <- read.table(file = paste(parpath,'immune_stimulated_expression_matrix.txt.gz',sep='/'), sep = "\t")

# Set up control object
ctrl <- CreateSeuratObject(counts = ctrl.data, project = "IMMUNE_CTRL", min.cells = 5)
ctrl$stim <- "CTRL"
ctrl <- subset(ctrl, subset = nFeature_RNA > 500)
ctrl <- NormalizeData(ctrl, verbose = FALSE)
ctrl <- FindVariableFeatures(ctrl, selection.method = "vst", nfeatures = 2000)

# Set up stimulated object
stim <- CreateSeuratObject(counts = stim.data, project = "IMMUNE_STIM", min.cells = 5)
stim$stim <- "STIM"
stim <- subset(stim, subset = nFeature_RNA > 500)
stim <- NormalizeData(stim, verbose = FALSE)
stim <- FindVariableFeatures(stim, selection.method = "vst", nfeatures = 2000)
