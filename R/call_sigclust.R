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
if (TRUE){
  args = commandArgs(TRUE)
  tmpname = args[1]
  inpath = paste(tmpname,"_in.mat",sep="")
  outpath = paste(tmpname,"_out.mat",sep="")
  outpath_r = paste(tmpname,'_out.RData',sep="")
}else{
  tmpname='/Volumes/SSD_Q/P_embedding/embed_rhesus_joint/sigclust_data'
  tmpname='/mnt/scratch/BV_embed/embed_rhesus_jointAngle/sigclust_data'
  inpath = paste(tmpname,'_in.mat',sep="")
  outpath = paste(tmpname,'_out.mat',sep="")
  outpath_r = paste(tmpname,'_out.RData',sep="")
}

# read in leaves to use as clusters
dat <- readMat(inpath)

sz = dat[["T.size"]] #size of tree marix
#sz = dat[["sz"]] #size of tree marix
T = matrix(dat[["T"]],nrow=sz[1],byrow=FALSE) # tree
label=dat[["label"]] #labels for clusters
X=dat[["X"]] 
nrand=unlist(dat[["nrand"]][1])
nsim=unlist(dat[["nsim"]][1])
pItem=unlist(dat[["pItem"]][1])
maxItem=unlist(dat[["maxItem"]][1])

icovest=3
#nrand=10
#pItem=0.05
#maxItem = 100

# loop through all split points
P=c()
CIr=c()
CI=c()
N=c()
for (it in 1:sz[1]){
  tic()
  print(paste('pair',it ,'/',sz[1]))
  # labels
  c1 = T[it,1]
  c2 = T[it,2]
  
  # selection
  sel1 = which(label %in% unlist(c1))
  sel2 = which(label %in% unlist(c2))
  
  n1 = min(max(ceiling(length(sel1) * pItem),2),maxItem)
  n2 = min(max(ceiling(length(sel2) * pItem),2),maxItem)
  #mn = min(n1,n2)
  #n1 = mn
  #n2 = mn
  
  N = rbind(N,c(n1,n2))
  
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
    res = sigclust(x,nsim=nsim,labflag=1,label=lab,icovest = icovest)
    
    # store
    cir = cbind(cir,res@simcindex)
    ci = c(ci,res@xcindex)
  }
  
  # get significance: smaller CI==stronger clustering (Liu 2008)
  p=sum(cir<mean(ci))/length(cir)
  
  # store
  P=c(P,p)
  CIr=rbind(CIr,colMeans(cir))
  CI=rbind(CI,ci)
  toc()
  
  foo=1
}

# save
print(paste("saving:",outpath))
writeMat(outpath,P=P,CI=CI,CIr=CIr,T=T,N=N)



