#! /usr/bin/Rscript

library("R.matlab")
library("bigtime")
# testing

# unfurl the command line argument
if (FALSE){
	args = commandArgs(TRUE)
	tmpname = args[1]
	inpath = paste(tmpname,"_in.mat",sep="")
	outpath = paste(tmpname,"_out.mat",sep="")
} else{
	# import the data
	# inpath = "/Users/ben/Desktop/test_glmnet/ke_VL06_2017-09-04_053_01-A_SE11a_rewStart_tmp.txt"
	# outpath = "/Users/ben/Desktop/test_glmnet/ke_VL06_2017-09-04_053_01-A_SE11a_rewStart_tmpres.txt"
	#dat = fromJSON(inpath)
  inpath="/Users/Ben/Desktop/test_ssm/data_in.mat"
  outpath="/Users/Ben/Desktop/test_ssm/data_out.mat"
}
dat <- readMat(inpath)

# extract the data
C=dat[["cd"]]
npose=dat[["npose"]]

# fit model
mdl = sparseVAR(C) # sparse VAR

foo=1
#dummy.call <- t(c("x", "y", "family", "options"))
#writeMat(outpath,Ball=Ball,Bmin=Bmin,B1se=B1se,B=B)
