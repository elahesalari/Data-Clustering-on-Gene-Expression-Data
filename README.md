# Data Clustering on Gene Expression Data

Understanding gene behavior is essential for studying cellular functions.  
Clustering gene expression data helps identify co-expressed genes, uncover unknown gene functions, and discover functional relationships between genes. Genes with similar expression patterns can be grouped into clusters to reveal biological insights.

---

## Dataset

The dataset used is the **Spellman dataset** (`Spellman.csv`), which contains gene expression measurements from *Saccharomyces cerevisiae* cell cultures synchronized at different cell cycle stages using a temperature-sensitive mutation (cdc15-2).  

- Contains **4381 transcripts** (rows 2 to 4382) measured at **23 time points** (columns).  
- Rows represent genes; columns represent time points (experiments).  
- Each cell value corresponds to the expression level (log ratio) of a gene at a specific time point.  
- The first column contains gene names; the first row is the header.

---

## Feature Extraction and Clustering

An **Auto-Encoder (AE)** neural network is used to reduce dimensionality by learning a compressed representation of the gene expression data in its latent space. This lower-dimensional representation helps improve clustering performance.

### Steps completed:

1. Constructed and trained an AE with **3 neurons** in the latent space on the dataset.  
2. Extracted latent features by feeding the data through the trained AE.  
3. Implemented **k-means clustering** with *k = 3* and *k = 4*, as well as a **Gaussian Mixture Model (GMM)** clustering method.  
4. Compared clustering results using the **Davies–Bouldin Index (DBI)** to evaluate cluster quality.  
5. Repeated the above steps using an AE with **5 neurons** in the latent space.


