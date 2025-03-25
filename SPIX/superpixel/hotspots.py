import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np
import numpy.typing as npt
from anndata import AnnData
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from .slic_segmentation import slic_segmentation


def find_hotspots(
    adata: AnnData,
    genes: list,
    max_spots: int = 100,
    depth_exclusion: int = 3, # TODO add radius search as well
    embedding = "X_embedding",
    compactness: float = 1,
    scaling: float = 0.3,
    max_iter: int = 1000,
    verbose: bool = True) -> AnnData:
    #-------------------------------------------------------------------------#
    # First lets get a gene set using sum of counts
    #-------------------------------------------------------------------------#
    gene_set = get_gene_order(adata, genes)
    #-------------------------------------------------------------------------#
    # Creating Spatial Graph or distance matrix
    #-------------------------------------------------------------------------#
    exclusion_zone = get_exclusion_zone(adata, depth_exclusion)
    #-------------------------------------------------------------------------#
    # Get initial indices from hotspots
    #-------------------------------------------------------------------------#
    hotspots_index = get_hotspot_index(
        adata,
        gene_set,
        exclusion_zone,
        max_spots)
    #-------------------------------------------------------------------------#
    # SLIC using hotspots
    # At the moment not sure how we should include the other points
    # I think adding a exclusion set so we don't include any other unwanted
    # coordinates could be useful.
    #-------------------------------------------------------------------------#
    if embedding not in adata.obsm:
        raise ValueError(f"Embedding '{embedding}' not found in adata.obsm.")
    else :
        embeddings = adata.obsm['X_embedding']
    clusters, combined_data = slic_segmentation(
        embeddings = embeddings,
        spatial_coord = adata.obsm['spatial'],
        n_segments = len(hotspots_index),
        compactness = compactness,
        scaling = scaling,
        index_selection = hotspots_index,
        max_iter = max_iter,
        verbose = verbose)
    
    return clusters, combined_data



def get_gene_order(
    adata: AnnData,
    genes: list):
    adata = adata[:, genes]
    gene_sum = adata.X.sum(axis = 1)
    gene_order = np.argsort(gene_sum)[::-1]
    return gene_order.tolist()

def get_exclusion_zone(
    adata : AnnData,
    depth_exclusion: int = 3):
    sq.gr.spatial_neighbors(adata, delaunay = True, coord_type = "generic")
    graph_distances = dijkstra(adata.obsp['spatial_connectivities'], directed=False)
    neighbors = [np.where(graph_distances[i] <= depth_exclusion)[0].tolist() for i in range(graph_distances.shape[0])]
    return neighbors
    

def get_hotspot_index(
    adata:AnnData,
    gene_set : list,
    exclusion_zone : list,
    max_spots: int = 100):
    initial_index = []
    for i in range(max_spots):
        expr_loc = gene_set.index(max(gene_set))
        initial_index.append(expr_loc)
        gene_set = [j for j in gene_set if j not in exclusion_zone[i]]
        if len(gene_set) == 0 or len(initial_index) >= max_spots:
            break
        else:
            continue
    return initial_index