import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np
import numpy.typing as npt
from anndata import AnnData
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from .slic_segmentation import slic_segmentation
from ..utils.utils import create_pseudo_centroids


def find_hotspots(
    adata: AnnData,
    genes: list,
    depth_exclusion: int = 3, # TODO add radius search as well
    embedding = "X_embedding",
    dimensions: list = list(range(30)),
    compactness: float = 1,
    scaling: float = 0.3,
    max_iter: int = 1000,
    add_name: str = 'Hotspots',
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
        gene_set,
        exclusion_zone)
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
        spatial_coords = adata.obsm['spatial'],
        n_segments = len(hotspots_index),
        compactness = compactness,
        scaling = scaling,
        index_selection = hotspots_index,
        max_iter = max_iter,
        verbose = verbose)
    # TODO clean this section up - relying on other functions but might be good
    # to have this in a more streamline manner. Maybe we should leave the 
    # the vesalius code style behind
    clusters_df = pd.DataFrame({
        'barcode': adata.obs.index,
        'Segment': clusters
    })

    
    tile_colors = pd.DataFrame(np.array(adata.obsm[embedding])[:, dimensions])
    tile_colors['barcode'] = adata.obs.index
    tiles = adata.uns['tiles'][adata.uns['tiles']['origin'] == 1]
    
    embeddings_df = pd.merge(tiles, tile_colors, on="barcode", how="right").dropna().reset_index(drop=True)
    embeddings_df = embeddings_df.drop(columns=['barcode',"x", "y", "origin"])
    embeddings_df.index = adata.obs.index
    pseudo_centroids = create_pseudo_centroids(
        embeddings_df,
        clusters_df,
        dimensions
    )
    adata.obsm['X_embedding_segment'] = pseudo_centroids
    adata.obsm['X_embedding_scaled_for_segment'] = combined_data
    
    adata.obs[add_name] = pd.Categorical(clusters_df['Segment'])
    return adata
    
    return adata



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
    gene_set : list,
    exclusion_zone : list):
    initial_index = []
    i = 0
    while len(gene_set) > 0:
        expr_loc = gene_set.index(max(gene_set))
        initial_index.append(expr_loc)
        gene_set = [j for j in gene_set if j not in exclusion_zone[i]]
        i += 1
        if len(gene_set) == 0:
            break
        else:
            continue
    return initial_index