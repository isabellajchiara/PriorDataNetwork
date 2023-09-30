library(AlphaSimR)

genMap <- readRDS("data/genMapSNPs.RData")
haplotypes <- readRDS("data/haplotypesSNPs.RData")

founderPop = newMapPop(genMap, 
                       haplotypes, 
                       inbred = FALSE, 
                       ploidy = 2L)

SP <- SimParam$new(founderPop)
SP$addTraitAEG(10, mean=8.8)
SP$setVarE(h2=0.25)

## randomly cross 200 parents 
Parents = newPop(founderPop)
F1 = randCross(Parents, 200)

## self and bulk F1 to form F2 ##

F2 = self(F1, nProgeny = 10)
F2 = setPheno(F2)

## select top 100 individuals from F2 bulk and self to form F3
F3Sel = selectInd(F2, 100, use="pheno", top=TRUE) 
F3 = self(F3Sel, nProgeny=30)
F3 = setPheno(F3)

##select top individuals within F3 families to form F4 ##

F4Sel = selectWithinFam(F3, 10, use="pheno", top=TRUE) 
F4 = self(F4Sel)
F4 = setPheno(F4)

## select top families from F4 to form F5 ##

F5Sel = selectFam(F4, 30, use="pheno", top=TRUE)
F5 = self(F5Sel)
F5 = setPheno(F5)


## select top families from F5 for PYTs ##

PYTSel = selectFam(F5, 16, use="pheno", top=TRUE) 
PYT = self(PYTSel, nProgeny = 2)
PYT = setPheno(PYT, reps=2)

newGen = self(PYT)
nGen=10
while (x < nGen){
  newGen = self(newGen)
  nGen = x+1
}

sample = sample(100:200,1)
npops = 10
y=1
for (n in 1:npops)  {
popname <- selectInd(newGen, 5)
  while (y < sample){
    popname = self(popname, nProgeny=30)
    y=y+1
    }
assign(paste0("pop",n), popname)
}

pops = list(pop1,pop2,pop3,pop4,pop5,pop6,pop7,pop8,pop9,pop10)

sample(100:200,1)
parents = selectInd(pops$y,5)
parents2 = selectInd(pops$z,5)
newpop = randcross2(parents, parents2)



