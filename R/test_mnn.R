#! /usr/bin/Rscript

library("batchelor")
library("R.matlab")


# unfurl the command line argument
if (TRUE){
	args = commandArgs(TRUE)
	tmpname = args[1]
	inpath = paste(tmpname,"_in.mat",sep="")
	outpath = paste(tmpname,"_out.mat",sep="")
	outpath_r = paste(tmpname,'_out.RData',sep="")
}else{
	tmpname='/Users/Ben/Desktop/test_limb2/mnn_dat'
	inpath = paste(tmpname,'_in.mat',sep="")
	outpath = paste(tmpname,'_out.mat',sep="")
	outpath_r = paste(tmpname,'_out.RData',sep="")
}

dat <- readMat(inpath)

X = dat[["X"]]
batch = dat[["batch"]]
d = dat[["d"]]
k = dat[["k"]]
nparallel = dat[["nparallel"]]

if (d==0){
	d=NA
}

if (nparallel > 1){
	print("parallel")
	p=BiocParallel::MulticoreParam(nparallel)
} else {
	print("serial")
	p=BiocParallel::SerialParam()
}

# run it
print("run fastMNN")
mnn.out <- fastMNN(X, batch=batch, d=d, k=k,cos.norm=FALSE,BPPARAM = p)
#X=t(X)
#mnn.out <- reducedMNN(X, batch=batch, k=k)
#mnn.out

# save stuff
writeMat(outpath, x=assay(mnn.out, "reconstructed"))
save("mnn.out",file=outpath_r)

# correction vector
#cor.exp <- assay(mnn.out)[id,]
#hist(cor.exp, xlab="Corrected expression for gene", col="grey80")
