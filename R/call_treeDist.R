#! /usr/bin/Rscript

library("R.matlab")
library("TreeDist")
library("ape")

# unfurl the command line argument
if (FALSE){
  args = commandArgs(TRUE)
  tmpname = args[1]
  inpath = paste(tmpname,"_in.mat",sep="")
  outpath = paste(tmpname,"_out.mat",sep="")
  outpath_r = paste(tmpname,'_out.RData',sep="")
}else{
  tmpname='/Volumes/SSD_Q/P_embedding/embed_rhesus_jointAngle_vcom_pcom_preAlignPose_pca/modularity/treeDist/treeDist'
  inpath = paste(tmpname,'_in.mat',sep="")
  outpath = paste(tmpname,'_out.mat',sep="")
  outpath_r = paste(tmpname,'_out.RData',sep="")
}

dat <- readMat(inpath)

treeList=dat[['treeList']]
ncmb=length(treeList)/2

for (ic in 1:1){
  ii1=ic
  ii2=ic+ncmb
  
  s=treeList[ii1]
  tree1 = read.tree(s)
  s=treeList[ii2]
  tree2 = read.tree(s)
}
#tree1 = dat[["tree1"]]
#tree2 = dat[["tree2"]]

#s='/Volumes/SSD_Q/P_embedding/embed_rhesus_jointAngle_vcom_pcom_preAlignPose_pca/modularity/treeDist/tree1.tree'
#tree1 = read.tree(s)
#s='/Volumes/SSD_Q/P_embedding/embed_rhesus_jointAngle_vcom_pcom_preAlignPose_pca/modularity/treeDist/tree2.tree'
#tree2 = read.tree(s)
