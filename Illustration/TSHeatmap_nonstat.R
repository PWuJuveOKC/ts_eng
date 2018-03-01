rm(list=ls())
library("gplots")
library(RColorBrewer)
df <- read.csv('../Data/nonstat_285_heat.csv',header=TRUE)
#rownames(df) <- df$name
# size = 20
# rownames(df) <- c(paste(rep("ECG",size),seq(1,size)),paste(rep("ART",size),seq(1,size)),
#                   paste(rep("CO2",size),seq(1,size)),paste(rep("PAP",size),seq(1,size)))

df <- df[ , -which(names(df) %in% c("name","Label"))]
features = colnames(df)
colnames(df) = substring(features,8,100)


#my_palette <- colorRampPalette(c("red","black","green"))(100)
pdf("../Output/heatmap_sample_nonstat.pdf", width=12, height=9)
heatmap.2(as.matrix(scale(df)),  Rowv=F,Colv=F, dendrogram ='none', trace='none',
          density.info="none",key=TRUE,keysize=0.6,offsetCol=0.2,
          margins=c(12,10),cexRow=1,cexCol=0.5,srtCol=45,
          labRow = ' ', col = redgreen(75),breaks=c(seq(-2,2,0.0533)),
  )

# par(cex=0.7)
# legend("topright",inset=c(0.01,-0.0),
#        legend = c('ECG','ART','CO2','PAP'),
#        col = c('#76EEC6','#EE6A50','#6495ED','#BF3EFF'),
#        lty= 2,
#        lwd = 12,
#        bty="n",
#        text.font=3
# )

dev.off()

