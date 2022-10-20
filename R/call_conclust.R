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

dat <- readMat(inpath)

X = dat[["X"]]
reps = dat[["reps"]]
pItem = dat[["pItem"]]
maxK = dat[["maxK"]]

# run consensus cluster
n = 3000
tmp=rbind(rnorm(n = n, mean = 0, sd = 1), rnorm(n = n, mean = 2, sd = 1))
#tmp = t(tmp)
label=as.vector(sample(1:2,n,replace=TRUE))

tmp=as.data.frame(t(X))
label=as.vector(sample(1:2,nrow(tmp),replace=TRUE))

mu <- 5
n <- 30
p <- 500
tmp <- matrix(rnorm(p*2*n),2*n,p)
tmp[1:n,1] <- tmp[1:n,1]+mu
tmp[(n+1):(2*n),1] <- tmp[(n+1):(2*n),1]-mu
label=cbind(rep(1,n),rep(2,n))

nsim <- 1000
nrep <- 1
icovest <- 3
pvalue = sigclust(tmp,nsim=nsim,labflag=1,label=label,icovest=icovest)

#sigclust plot
plot(pvalue)


ress = sigclust(tmp,50,labflag = 1,label = label)

tic("M3C")
res = M3C(tmp,cores=2, iters=25, pItem=0.2)
toc()

# read in leaves to use as clusters
tmp=readMat('/Users/Ben/Desktop/test_conclust/rand.mat')
T = tmp[["T"]]
sz = tmp[["sz"]]
m=matrix(T,nrow=sz[1],byrow=FALSE)


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
