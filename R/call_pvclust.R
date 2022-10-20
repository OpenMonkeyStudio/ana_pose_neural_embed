#! /usr/bin/Rscript

#library(ConsensusClusterPlus)
library(R.matlab)
#library(cola)
#library(M3C)
library(sigclust)
library(tictoc)

#suppressPackageStartupMessages(library(ConsensusClusterPlus))
suppressPackageStartupMessages(library(R.matlab))
#suppressPackageStartupMessages(library(cola))
#suppressPackageStartupMessages(library(M3C))

# unfurl the command line argument
if (FALSE){
  args = commandArgs(TRUE)
  tmpname = args[1]
  inpath = paste(tmpname,"_in.mat",sep="")
  outpath = paste(tmpname,"_out.mat",sep="")
  outpath_r = paste(tmpname,'_out.RData',sep="")
}else{
  tmpname='/Users/Ben/Desktop/test_sigclust/data'
  inpath = paste(tmpname,'_in.mat',sep="")
  outpath = paste(tmpname,'_out.mat',sep="")
  outpath_r = paste(tmpname,'_out.RData',sep="")
}

# read in leaves to use as clusters
dat <- readMat(inpath)

X=dat[["X"]] 
method.hclust=dat[["method_hclust"]] 
method.dist=dat[["method_dist"]]
nboot=dat[["nboot"]]
nparallel=dat[["nparallel"]]
r=dat[["r"]]

if (nparallel<2){
  parallel=FALSE
}else{
  parallel = nparallel
}

# have to take transpose... why??
res = pvclust(X,method.hclust = method.hclust,
              method.dist = method.dist,
              nboot=nboot,
              parallel = parallel,
              )


sz = dat[["sz"]] #size of tree marix
T = matrix(dat[["T"]],nrow=sz[1],byrow=FALSE) # tree
label=dat[["label"]] #labels for clusters
X=dat[["X"]] 
nrand=unlist(dat[["nrand"]][1])
nsim=unlist(dat[["nsim"]][1])
pItem=unlist(dat[["pItem"]][1])

icovest=3

# loop through all split points
P=c()
CIr=c()
CI=c()
for (it in 1:sz[1]){
  tic()
  print(paste('pair',it ,'/',sz[1]))
  # labels
  c1 = T[it,1]
  c2 = T[it,2]
  
  # selection
  sel1 = which(label %in% unlist(c1))
  sel2 = which(label %in% unlist(c2))
  
  n1 = min(ceiling(length(sel1) * pItem),2)
  n2 = min(ceiling(length(sel2) * pItem),2)
  #mn = min(n1,n2)
  #n1 = mn
  #n2 = mn
  
  # run multiple subsampled permutations to get p-value
  cir=c()
  ci=c()
  
  for (ir in 1:nrand){
    # perm index
    s1 = sample(sel1,n1,replace=FALSE)
    s2 = sample(sel2,n2,replace=FALSE)
    sel = c(s1,s2)
    
    # data
    x = X[sel,]
    lab=rep(1,length(sel))
    lab[sel %in% s2] = 2
    
    # run
    #res = sigclust(x,nsim=nsim,labflag=1,label=lab,icovest=icovest)
    res = sigclust(x,nsim=nsim,labflag=0,label=lab,icovest = icovest)
    
    # store
    CIr = c(CIr,res@simcindex)
    CI = c(CI,res@xcindex)
  }
  
  
  # get significance
  p=sum(cir>mean(ci))/length(cir)
  
  # store
  P=c(P,p)
  CIr=c(CIr,mean(cir))
  CI=c(CI,mean(ci))
  foo=1
  toc()
}



