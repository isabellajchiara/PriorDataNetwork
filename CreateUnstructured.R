library(AlphaSimR)
library(writexl)
library(devtools)
library(dplyr)
library(tidyverse)
library(ggplot2)
library(cluster)
library(factoextra)
library(ggpubr)


setwd("/Users/justin_folders/Desktop/Isabella_McGill_Files/GEBVs")

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
F1 = randCross(Parents, 200)

## self and bulk F1 to form F2 ##

F2 = self(F1, nProgeny = 4)
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

pops = list()
npops = 3 # create a random number of subpopulations between 3 and 7
selfGen = sample(5:7,1) # self the subpops for a random number of Gen between 50 and 100
for (n in 1:npops)  {
  popname <- selectInd(newGen, 128)
  c = 1
  while (c<nRand){
  popname <- randCross(popname,20)
  c = c +1
  }
  y = 1
  while (y < selfGen){
    popname = self(popname, nProgeny=3)
    y=y+1
  }
  pops[[n]] = assign(paste0("pop",n), popname)
}

cat("finished creating subpops")


newpoplist = list()
nMigrations = npops #select a random number of intermigrations
g = 10
for (m in 1:nMigrations){
  y = sample(1:npops,1) #randomly choose pop1
  z = sample(1:npops,1) #randomly choose pop2
  parents = selectInd(pops[[y]],30) #select migrating parents
  parents2 = selectInd(pops[[z]],30) #select migrating parents  
  newpop = hybridCross(parents, parents2,crossPlan="testcross") #cross migrating parents 
  g = 1
  while (g < nGen){
    newpop = self(newpop, nProgeny = 1)
    g = g+1
  }
  newpoplist[[m]] = assign(paste0("newPop",m), newpop) #collect new pops in a list
}

cat("finished migrations")

trainX = list()
trainY = list()
for (z in 1:npops){
  getData = newpoplist[[z]]
  geno = pullSegSiteGeno(getData)
  pheno = pheno(getData)
  trainX[[z]] = assign(paste0("genopop",z),geno)
  trainY[[z]] = assign(paste0("phenopop",z),pheno)
}

cat("finished pulling pop data")


trainingGeno = as.data.frame(do.call(rbind, trainX))
trainingPheno = as.data.frame(do.call(rbind,trainY))

realData = readRDS("haplotypesSNPs.RData")
haplo = as.data.frame(do.call(cbind, realData))
colnames(haplo) = paste0("ID",1:ncol(haplo))
colnames(trainingGeno) = paste0("ID",1:ncol(trainingGeno))

M = rbind(trainingGeno, haplo)


#visualize
newgeno <- trainingGeno %>%  select(where(~ n_distinct(.) > 1))
newgeno = scale(newgeno)
colnames(newgeno) =NULL
rownames(newgeno) = c(paste0("geno",1:nrow(newgeno)))

# optimize K
PCAgeno <- prcomp(newgeno, center=TRUE, scale=TRUE) ##take out categorical columns##
PCAselected = as.data.frame(-PCAgeno$x[,1:3])
silhouette <- fviz_nbclust(PCAselected, kmeans, method = 'silhouette')
kvalues <- silhouette$data ##largest value tells how many clusters are optimal ##
kvalues <- kvalues[order(-kvalues$y),]
k=as.numeric(kvalues[1,1])

# apply kmeans function and visualize
k2 <- kmeans(newgeno, centers = k, nstart = 25)
fviz_cluster(k2, data = newgeno,geom="point",ggtheme=theme_minimal(), ellipse = TRUE,
             ellipse.type = "convex",
             ellipse.level = 0.05,
             ellipse.alpha = 0.2,
             main = paste0("unstructured genotypes"))


