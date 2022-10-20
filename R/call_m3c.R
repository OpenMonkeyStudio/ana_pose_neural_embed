#! /usr/bin/Rscript

#library(ConsensusClusterPlus)
library(R.matlab)
#library(cola)
library(M3C)
library(sigclust)

#suppressPackageStartupMessages(library(ConsensusClusterPlus))
suppressPackageStartupMessages(library(R.matlab))
#suppressPackageStartupMessages(library(cola))
suppressPackageStartupMessages(library(M3C))

# unfurl the command line argument
if (FALSE){
  args = commandArgs(TRUE)
  tmpname = args[1]
  inpath = paste(tmpname,"_in.mat",sep="")
  outpath = paste(tmpname,"_out.mat",sep="")
  outpath_r = paste(tmpname,'_out.RData',sep="")
}else{
  tmpname='/Users/Ben/Desktop/test_conclust/pvclust_dat'
  inpath = paste(tmpname,'_in.mat',sep="")
  outpath = paste(tmpname,'_out.mat',sep="")
  outpath_r = paste(tmpname,'_out.RData',sep="")
}

# read in leaves to use as clusters
dat <- readMat(inpath)
#dat=readMat('/Users/Ben/Desktop/test_conclust/rand.mat')

sz = dat[["sz"]] #size of tree marix
T = matrix(dat[["T"]],nrow=sz[1],byrow=FALSE) # tree
labels=dat[["label"]] #labels for clusters
X=dat[["X"]] 
nrand=dat[["nrand"]]
nsim=dat[["nsim"]]
pItem=dat[["pItem"]]

# loop through all split points
for (it in 1:sz[1]){
  # labels
  c1 = T[it,1]
  c2 = T[it,2]
  
  # selection
  sel1 = which(label %in% c1)
  sel2 = which(label %in% c2)
  
  n1 = ceil(length(sel1) * pItem)
  n2 = ceil(length(sel2) * pItem)
  mn = min(n1,n2)
  n1 = mn
  n2 = mn

  # run multiple subsampled permutations to get p-value
  CIr=c()
  CI=c()
  
  for (ir in 1:nrand){
    # perm index
    s1 = sel1( sample(sel1,n1,replace=FALSE) )
    s2 = sel2( sample(sel2,n2,replace=FALSE) )
    sel = c(s1,s2)
    
    # data
    x = X[sel,]
    lab=rep(1,length(sel))
    lab(sel %in% s2) = 2
    
    # run
    res = sigclust(tmp,nsim=nsim,labflag=1,label=lab,icovest=icovest)
    
    # store
    CIr = c(CIr,res@simcindex)
    CI = c(CI,res@xcindex)
  }

  
  # get significance
  
  # store
  
  foo=1
}


# call
if (FALSE){
  #result <- pvclust(X, method.dist="euclidean", method.hclust="ward", nboot=100)
  results = ConsensusClusterPlus(X,maxK=maxK,reps=reps,pItem=pItem,pFeature=1,clusterAlg="hc",distance="euclidean",verbose=TRUE)
  
  # prep for matlab
  for (ii in 2:maxK){
    outpath_tmp=paste(tmpname,'_out',ii, '.mat',sep="")
    c = results[[ii]][["consensusClass"]];
    m = results[[ii]][["consensusTree"]]$merge;
    ord = results[[ii]][["consensusTree"]]$order;
    h = results[[ii]][["consensusTree"]]$height;
    
    writeMat(outpath_tmp, consensusClass=c,height=h,order=ord,merge=m)
  }
  
  ape::write.tree(dendro, file='/Users/Ben/Desktop/test_conclust/filename.txt')
  
  # save for R
  save("results",file=outpath_r)
}
