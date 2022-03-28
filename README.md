# Data-Clustering-on-Gene-Expression-Data

The knowledge on the behavior of genes is necessary to learn 
the nature of cellular functions. Most data mining algorithms,
developed for microarray gene expression data, deal with the 
difficulty of clustering. Cluster analysis of gene expression data 
helps to identify co-expressed genes. Analysis of these data sets 
reveals genes of unknown functions and the discovery of 
functional relationships between genes. Co-expressed genes can 
be grouped into clusters, based on their expression patterns of 
genes.

<b>Dataset:</b>
<br/>
The Spellman dataset (Spellman.csv) provides the gene 
expression data measured in Saccharomyces cerevisiae cell 
cultures that have been synchronized at different points of the 
cell cycle by using a temperature-sensitive mutation (cdc15-2), 
which arrests cells late in mitosis at the restrictive temperature 
(it can cause heat-shock). This dataset has 4381 transcripts (rows 
2:4382) measured at 23 time points (row 1). In the data matrix,
rows represent genes and columns correspond to time points
(experiments). So, each value in row i and column j corresponds to the expression level (log ratio) of gene gi in time point tj. The 
first column is the name of gene, and the first row is header of 
data.

<b>Part A: Feature Extraction and Clustering</b>
<br/>
An auto-encoder (AE) is a neural network that learns to copy its input to 
its output. One of the main usage of AEs is to reduce the dimension by 
extracting meaningful features in the latent space (code layer). 
Representing data in a lower-dimensional space can improve 
performance on different tasks, such as classification and clustering.
<br/>
  1. First, you should construct an AE with 3 neurons in the latent 
  space and train the network with the given dataset.
  2. Then, feed the data to the network and extract features from the 
  latent space.
  3. Implement k-means (with k=3 and 4) and Gaussian mixture model 
  (GMM) clustering methods and compare the results according to 
  the Davies–Bouldin index (DBI) criteria.
  4. Repeat the mentioned steps for network with 5 neurons in the 
  latent space.

<b>Part B: Gene Ontology (GO)</b>
<br/>
According to the DBI criteria, select the best result from part A 
and determine what information can be extracted from each
cluster. Use gene ontology (GO) for each identified cluster:
<br/>
  • Go to the g:Profiler website, http://biit.cs.ut.ee/gprofiler/gost .On
  the left box, enter your cluster gene names (whitespace-separated) <br/>
  • For options, choosing Saccharomyces cerevisiae from organism
  box. <br/>
  • Click Run query button. <br/>
  • The results are sorted by p-values in ascending order. Draw a table 
  to list the top 3 GO categories, showing the Term-name, Term-ID, 
  and p-value in each column.



