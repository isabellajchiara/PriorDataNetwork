library(AlphaSimR)
library(writexl)

genMap <- readRDS("genMapSNPs.RData")
haplotypes <- readRDS("haplotypesSNPs.RData")

founderPop = newMapPop(genMap, 
                       haplotypes, 
                       inbred = FALSE, 
                       ploidy = 2L)

SP <- SimParam$new(founderPop)
SP$addTraitAEG(10, mean=8.8)
SP$setVarE(h2=0.25)
SP$setSexes("yes_rand")

## randomly cross 200 parents 
Parents = newPop(founderPop)
F1 = randCross(Parents, 200, nProgeny=30)

## self and bulk F1 to form F2 ##

F2 = self(F1, nProgeny = 15)
F2 = setPheno(F2)

## select top 100 individuals from F2 bulk and self to form F3
F3Sel = selectInd(F2, 100, use="pheno", top=TRUE) 
F3 = self(F3Sel, nProgeny=15)
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
PYT = self(PYTSel, nProgeny = 4)
PYT = setPheno(PYT, reps=2)

# We will now advance elites to create an unstructured population

cat("finished initial pop")

newGen = self(PYT)

nGen= sample(10:20, 1) # advance a random number of generations between 20 and 50
x=1
while (x < nGen){
  newGen = self(newGen, nProgeny=1)
  x = x+1
}

cat("finished selfing initial Elites")


pops = list()
npops = sample(3:7,1) # create a random number of subpopulations between 3 and 7
selfGen = sample(5:10,1) # self the subpops for a random number of Gen between 50 and 100
for (n in 1:npops)  {
  y=1
  popname <- selectInd(newGen, 5)
  while (y < selfGen){
    popname = self(popname, nProgeny=4)
    y=y+1
  }
  pops[[n]] = assign(paste0("pop",n), popname)
}

cat("finished creating subpops")


newpoplist = list()
nMigrations = sample(3:5,1) #select a random number of intermigrations
nGen = sample(3:5,1)
for (m in 1:nMigrations){
  y = sample(1:npops,1) #randomly choose pop1
  z = sample(1:npops,1) #randomly choose pop2
  parents = selectInd(pops[[y]],3) #select migrating parents
  parents2 = selectInd(pops[[z]],3) #select migrating parents  
  newpop = hybridCross(parents, parents2,crossPlan="testcross") #cross migrating parents 
  advance = self(newpop, nProgeny=15)
  g = 1
  while (g < nGen){
    advance = self(advance, nProgeny=3)
    g = g+1
  }
  newpoplist[[m]] = assign(paste0("newPop",m), advance) #collect new pops in a list
}

cat("finished intermigrations and advancing")

trainX = list()
trainY = list()
for (z in 1:length(newpoplist)){
  getData = newpoplist[[z]]
  geno = pullSegSiteGeno(getData)
  pheno = pheno(getData)
  trainX[[z]] = assign(paste0("genopop",z),geno)
  trainY[[z]] = assign(paste0("phenopop",z),pheno)
  z = z+1
}

cat("finished pulling pop data")


trainingGeno = as.data.frame(do.call(rbind, trainX))
trainingPheno = as.data.frame(do.call(rbind,trainY))

cat("writing files")

write_xlsx(trainingGeno,"unstructuredGeno.xlsx")
write_xlsx(trainingPheno,"unstructuredPheno.xlsx")

cat("finished")


