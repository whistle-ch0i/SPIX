import numpy as np
import pandas as pd
import scanpy as sc
import anndata as an


def sum_gene_set(adata : AnnData,
    gene_set : list,
    threshold: float = 0.5,
    k : int = 500):
    
    sub_adata = adata[:,gene_set]
    counts = sub_adata.X
    
    cell_sum = np.sum(counts, axis = 1)
    cell_loc = np.argsort(cell_sum)[::-1]
    
    
    return cell_sum
    
    
    