import squidpy as sq
import pandas as pd
import numpy as np
import math

from anndata import AnnData
from scipy.sparse.csgraph import dijkstra
from .slic_segmentation import slic_segmentation
from typing import Union, Optional


def find_hotspots(
    adata: AnnData,
    genes: list,
    depth_exclusion: int = 3,
    top_hits: int = 5,
    min_links: Optional[Union[None, int]] = None,
    threshold:float = 0.7, 
    embedding :str = "X_embedding",
    compactness: float = 1,
    scaling: float = 0.3,
    max_iter: int = 1000,
    weighted: bool = False,
    add_name: str = 'Hotspots',
    verbose: bool = True) -> AnnData:
    """
    Generate super pixels from gene set - Genetic Hotspots
    
    Parameters
    ----------
    adata : AnnData
        AnnData containing counts and coordinates (minimal input)
    genes : list
        Gene list 
    depth_exclusion : int, optional (default = 3)
        If cells with high gene expression are within range of previous hotspot,
        exclude from new hotspot list
    top_hits : int, optional (default = 5)
        How many hotspots to use as initial vertices for interconnected vertex search
    min_links : Union[None, int], optional (default = None)
        Minimum number of edges a vertex should share with initial vertex set. 
        If None, will take half of the number of top hits
    threshold : float, optional (default = 0.7)
        Correlation Threshold between vertices - Vertices below threshould will be pruned if below
    embedding : str, optional (default = 'X_embedding)
        Which embedding value to use during SLIC segementation
    compactness : float, optional (default=1.0)
        Compactness parameter influencing the importance of spatial proximity in clustering.
    scaling : float, optional (default = 0.3)
        Scaling factor for spatial coordinates.
    max_iter : int, optional (default = 1000)
        Maximum number of iterations for convergence in initial cluster selection.
    weighted: bool, optional (default = False)
        Use a weighted SLIC segmentation approach
        Weights are proportional to rank of hotspot - stronger hotspots receive stronger weight
    add_name: str, optional (default = 'Hotspots')
        hotspot label to use 
    verbose : bool, optional (default=True)
        Whether to display progress messages.
    
    Returns
    -------
    An AnnData object with gene hotspots 
    """
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
    hotspot_indices = get_hotspot_index(
        gene_set,
        exclusion_zone)
    hotspot_locs = assign_hotspot(hotspot_indices, exclusion_zone)
    #-------------------------------------------------------------------------#
    # Weighted kmeans - giving higher weight to points in top hotspots
    # TODO: thorough testing 
    #-------------------------------------------------------------------------#
    if weighted:
        weights = weight_hotspots(hotspot_locs)
    else:
        weights = np.ones(len(hotspot_locs))
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
    clusters, _ = slic_segmentation(
        embeddings = embeddings,
        spatial_coords = adata.obsm['spatial'],
        n_segments = len(hotspot_indices),
        compactness = compactness,
        scaling = scaling,
        index_selection = hotspot_indices,
        weights = weights,
        max_iter = max_iter,
        verbose = verbose)
    #-------------------------------------------------------------------------#
    # TODO: check if we need to get embedding values for this. 
    # Intuation is no since we just want to find hotspots
    #-------------------------------------------------------------------------#
    clusters_df = pd.DataFrame({
        'barcode': adata.obs.index,
        'Segment': clusters
    })
    #-------------------------------------------------------------------------#
    # Get correlation scores between each hotspot
    # and use graph connectivity to define which hotspots to keep
    # TODO: find potential ways to reduce sensitivity to corr threshold
    #-------------------------------------------------------------------------#
    score_matrix = score_hotspots(clusters_df, adata)
    if min_links is None:
        min_links = math.ceil(top_hits / 2)
        
    clusters_df = graph_grouping(
        score_matrix,
        clusters_df,
        hotspot_indices[:top_hits],
        threshold,
        min_links = min_links)
    #-------------------------------------------------------------------------#
    # Rebuild adata and return
    # TODO: refactor for clarity - don't need all of these values in 
    # final version
    #-------------------------------------------------------------------------#
    adata.obs[f'{add_name}_locations_init'] = hotspot_locs
    adata.obs[f'{add_name}_locations'] = pd.Categorical(clusters_df['Hotspots'])
    adata.obs[f'{add_name}_all_spix'] = pd.Categorical(clusters_df['Segment'])
    return adata



def get_gene_order(
    adata: AnnData,
    genes: list) -> list:
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
    idx = 0
    while len(gene_set) > 0:
        expr_loc = gene_set[0]
        initial_index.append(expr_loc)
        gene_set = [j for j in gene_set if j not in exclusion_zone[expr_loc]]
        idx += 1
        if len(gene_set) == 0:
            break
        else:
            continue
    return initial_index

def assign_hotspot(initial_index, exclusion_zone):
    hotspot_locs = np.zeros(len(exclusion_zone))
    initial_index = initial_index[::-1]
    for idx, val in enumerate(initial_index):
        hotspot_locs[exclusion_zone[val]] = idx
    return hotspot_locs

def weight_hotspots(hotspot_locs):
    hotspot_weights = np.exp(-hotspot_locs) + 1
    return hotspot_weights 

def score_hotspots(cluster_df, adata):
    hotspot_indices = sorted(set(cluster_df['Segment']))
    n_hotspots = len(hotspot_indices)
    n_genes = adata.X.shape[1] # might need to change this for variable genes
    mean_vectors = np.zeros((n_hotspots,n_genes))
    for segment in hotspot_indices:
        barcodes = cluster_df.loc[cluster_df['Segment'] == segment, 'barcode']
        data = adata[barcodes, :].X
        mean_vectors[segment] = data.mean(axis=0)
    # For now we will use this method.
    score_matrix = np.corrcoef(mean_vectors)
    return score_matrix




def graph_grouping(
    corr_matrix,
    clusters_df,
    top_hits,
    threshold=0.975,
    include_initial=True,
    min_links = 2):
    # Extract the initial set of nodes
    initial_indices = clusters_df.loc[top_hits, 'Segment'].tolist()

    n = corr_matrix.shape[0]
    connected_nodes = set()

    for node in range(n):
        
        if not include_initial and node in initial_indices:
            continue

        
        connections = [corr_matrix[node, i] for i in initial_indices]
        strong_links = sum(1 for c in connections if c >= threshold)

        if strong_links >= min_links:
            connected_nodes.add(node)

    
    segment_mapping = {segment_id: f'Hotspot_{i}' for i, segment_id in enumerate(sorted(connected_nodes))}
    clusters_df['Hotspots'] = "Background"

    for segment_id, label in segment_mapping.items():
        clusters_df.loc[clusters_df['Segment'] == segment_id, 'Hotspots'] = label

    return clusters_df
