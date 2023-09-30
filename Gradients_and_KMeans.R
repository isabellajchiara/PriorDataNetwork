#These plots are ideally created in R studio
#Visualize clusters in pop based on genotype
#Visualize distances based on genotype

library(devtools)
library(dplyr)
library(tidyverse)
library(ggplot2)
library(cluster)
library(factoextra)
library(ggpubr)


#prepare DF
M = as.data.frame(pullSegSiteGeno(F2))
newgeno <- M %>%  select(where(~ n_distinct(.) > 1))
newgeno = scale(newgeno)
colnames(newgeno) =NULL

# optimize K
PCAgeno <- prcomp(newgeno, center=TRUE, scale=TRUE) ##take out categorical columns##
PCAselected = as.data.frame(-PCAgeno$x[,1:3])
silhouette <- fviz_nbclust(PCAselected, kmeans, method = 'silhouette')
kvalues <- silhouette$data ##largest value tells how many clusters are optimal ##
kvalues <- kvalues[order(-kvalues$y),]
k=as.numeric(kvalues[1,1])

# apply kmeans function and visualize
k2 <- kmeans(newgeno, centers = k, nstart = 25)
fviz_cluster(k2, data = newgeno)

#visualize disstances on genotype matrix
distance <- get_dist(newgeno)
fviz_dist(distance, gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))

  
