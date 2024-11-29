import json
import requests
import numpy as np
import pandas as pd
import io
import logging
from typing import List, Dict, Optional, Tuple
import seaborn as sns
import matplotlib.pyplot as plt

# Configure a module-level logger
logger = logging.getLogger('SPIX.enrichment')
logger.setLevel(logging.INFO)  # Default level
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Prevent adding multiple handlers
if not logger.handlers:
    logger.addHandler(handler)

# Disable propagation to avoid duplicate logs
logger.propagate = False

def set_logger_verbosity(verbose: bool) -> None:
    """
    Set the logger's verbosity.

    Parameters
    ----------
    verbose : bool
        If True, set logger to DEBUG level. Otherwise, set to WARNING.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)


def get_gene_array(filename: str, verbose: bool = True) -> List[str]:
    """
    Load a list of genes from a text file.

    Parameters
    ----------
    filename : str
        Path to the gene list file.
    verbose : bool, optional
        If True, enable detailed logging. Default is True.

    Returns
    -------
    List[str]
        List of gene symbols.
    """
    set_logger_verbosity(verbose)
    try:
        genes = np.loadtxt(filename, dtype='object', unpack=True)
        genes = genes.tolist()
        logger.debug(f"Loaded {len(genes)} genes from {filename}.")
        return genes
    except Exception as e:
        logger.error(f"Error loading genes from {filename}: {e}")
        raise


def get_background_array(filename: str, verbose: bool = True) -> List[str]:
    """
    Load a list of background genes from a text file.

    Parameters
    ----------
    filename : str
        Path to the background gene list file.
    verbose : bool, optional
        If True, enable detailed logging. Default is True.

    Returns
    -------
    List[str]
        List of background gene symbols.
    """
    set_logger_verbosity(verbose)
    try:
        background = np.loadtxt(filename, dtype='object', unpack=True)
        background = background.tolist()
        logger.debug(f"Loaded {len(background)} background genes from {filename}.")
        return background
    except Exception as e:
        logger.error(f"Error loading background genes from {filename}: {e}")
        raise


def write_enrichment_file(
    genes: List[str],
    database: str,
    export_filename: str,
    verbose: bool = True
) -> None:
    """
    Perform enrichment analysis using Enrichr and save the results to a file.

    Parameters
    ----------
    genes : List[str]
        List of gene symbols to analyze.
    database : str
        Name of the Enrichr database to use (e.g., "KEGG_2021_Human").
    export_filename : str
        Base name for the exported enrichment results file.
    verbose : bool, optional
        If True, enable detailed logging. Default is True.
    """
    set_logger_verbosity(verbose)
    ENRICHR_ADDLIST_URL = 'https://maayanlab.cloud/Enrichr/addList'
    ENRICHR_EXPORT_URL = 'https://maayanlab.cloud/Enrichr/export'
    description = 'Example gene list'

    # Submit the gene list to Enrichr
    logger.info(f"Submitting gene list to Enrichr for database: {database}")
    payload = {
        'list': (None, '\n'.join(genes)),
        'description': (None, description)
    }

    response = requests.post(ENRICHR_ADDLIST_URL, files=payload)
    if not response.ok:
        logger.error(f"Error adding gene list to Enrichr: {response.text}")
        raise Exception('Error analyzing gene list')

    data = json.loads(response.text)
    logger.debug(f"Enrichr Add List Response: {data}")

    user_list_id = data.get('userListId')
    if not user_list_id:
        logger.error("No userListId found in Enrichr response.")
        raise Exception('No userListId found in Enrichr response.')

    # Prepare the export request
    export_url = f"{ENRICHR_EXPORT_URL}?userListId={user_list_id}&filename={export_filename}&backgroundType={database}"
    logger.info(f"Export URL: {export_url}")

    # Request the enrichment results
    response = requests.get(export_url, stream=True)
    if not response.ok:
        logger.error(f"Error fetching enrichment results: {response.text}")
        raise Exception('Error fetching enrichment results')

    # Save the results to a file
    with open(f"{export_filename}.txt", 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    logger.info(f"Enrichment results saved to {export_filename}.txt")


def write_background_enrichment_file(
    genes: List[str],
    background: List[str],
    database: str,
    export_filename: str,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Perform background enrichment analysis using SpeedRICHr and save the results to a file.

    Parameters
    ----------
    genes : List[str]
        List of gene symbols to analyze.
    background : List[str]
        List of background gene symbols.
    database : str
        Name of the SpeedRICHr background database to use (e.g., "GO_Biological_Process_2023").
    export_filename : str
        Base name for the exported enrichment results file.
    verbose : bool, optional
        If True, enable detailed logging. Default is True.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the enrichment results.
    """
    set_logger_verbosity(verbose)
    base_url = "https://maayanlab.cloud/speedrichr"
    description = "Sample gene set with background"

    # Add gene list
    logger.info("Adding gene list to SpeedRICHr.")
    res = requests.post(
        f"{base_url}/api/addList",
        files={
            'list': (None, '\n'.join(genes)),
            'description': (None, description),
        }
    )
    if not res.ok:
        logger.error(f"Failed to add gene list: {res.text}")
        raise Exception(f"Failed to add gene list: {res.text}")

    userlist_response = res.json()
    logger.debug(f"User List Response: {userlist_response}")

    # Add background
    logger.info("Adding background gene list to SpeedRICHr.")
    res = requests.post(
        f"{base_url}/api/addbackground",
        data={'background': '\n'.join(background)}
    )
    if not res.ok:
        logger.error(f"Failed to add background: {res.text}")
        raise Exception(f"Failed to add background: {res.text}")

    background_response = res.json()
    logger.debug(f"Background Response: {background_response}")

    # Perform background enrichment
    logger.info("Performing background enrichment analysis.")
    res = requests.post(
        f"{base_url}/api/backgroundenrich",
        data={
            'userListId': userlist_response['userListId'],
            'backgroundid': background_response['backgroundid'],
            'backgroundType': database,
        }
    )
    if not res.ok:
        logger.error(f"Failed to perform background enrichment: {res.text}")
        raise Exception(f"Failed to perform background enrichment: {res.text}")

    results = res.json()
    logger.info("Background enrichment analysis completed.")

    # Create DataFrame from results
    df = pd.DataFrame(results)
    df['Term'] = df[database].apply(lambda x: x[1])
    df['P-value'] = df[database].apply(lambda x: x[2])
    df['Genes'] = df[database].apply(lambda x: ';'.join(x[5]))
    df['Adjusted P-value'] = df[database].apply(lambda x: x[6])

    # Remove the original column
    df.drop(columns=[database], inplace=True)

    # Save the results to a file
    df.to_csv(f"{export_filename}.txt", sep='\t', index=False)
    logger.info(f"Background enrichment results saved to {export_filename}.txt")

    return df


def get_enrichment_dataframe(
    genes: List[str],
    database: str,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Analyze a gene set using Enrichr and return the enrichment results as a pandas DataFrame.

    Parameters
    ----------
    genes : List[str]
        List of gene symbols.
    database : str
        Name of the Enrichr database to use (e.g., "KEGG_2021_Human").
    verbose : bool, optional
        If True, enable detailed logging. Default is True.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the enrichment results.

    Example
    -------
    enrichment_df = get_enrichment_dataframe(
        genes=["BRCA1", "TP53", "EGFR"],
        database="KEGG_2021_Human"
    )
    """
    set_logger_verbosity(verbose)
    ENRICHR_ADDLIST_URL = 'https://maayanlab.cloud/Enrichr/addList'
    ENRICHR_EXPORT_URL = 'https://maayanlab.cloud/Enrichr/export'
    description = 'Gene list for enrichment analysis'

    # Submit the gene list to Enrichr
    logger.info(f"Submitting gene list to Enrichr for database: {database}")
    payload = {
        'list': (None, '\n'.join(genes)),
        'description': (None, description)
    }

    response = requests.post(ENRICHR_ADDLIST_URL, files=payload)
    if not response.ok:
        logger.error(f"Error adding gene list to Enrichr: {response.text}")
        raise Exception(f'Error adding gene list to Enrichr: {response.text}')

    data = response.json()
    logger.debug(f"Enrichr Add List Response: {data}")

    user_list_id = data.get('userListId')
    if not user_list_id:
        logger.error("No userListId found in Enrichr response.")
        raise Exception('No userListId found in Enrichr response.')

    # Prepare the export request
    export_url = f"{ENRICHR_EXPORT_URL}?userListId={user_list_id}&filename=enrichment_results&backgroundType={database}"
    logger.info(f"Export URL: {export_url}")

    # Request the enrichment results
    response = requests.get(export_url, stream=True)
    if not response.ok:
        logger.error(f"Error fetching enrichment results: {response.text}")
        raise Exception(f'Error fetching enrichment results: {response.text}')

    # Read the response content
    content = response.content.decode('utf-8')

    # Use StringIO to read the content into pandas as if it were a file
    df = pd.read_csv(io.StringIO(content), sep='\t')
    logger.debug(f"Enrichment results loaded into DataFrame with {df.shape[0]} terms.")

    return df


def generate_enrichment_results(
    gene_groups: Dict[str, List[str]],
    databases: List[str],
    verbose: bool = True
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Perform enrichment analysis for multiple gene groups across multiple databases.

    Parameters
    ----------
    gene_groups : Dict[str, List[str]]
        Dictionary where keys are group names and values are lists of gene symbols.
    databases : List[str]
        List of database names to perform enrichment analysis.
    verbose : bool, optional
        If True, enable detailed logging. Default is True.

    Returns
    -------
    Dict[str, Dict[str, pd.DataFrame]]
        Nested dictionary containing enrichment results.
        The first key is the group name, the second key is the database name,
        and the value is the corresponding enrichment DataFrame.
    """
    set_logger_verbosity(verbose)
    enrichment_results = {}

    # Perform enrichment analysis
    for group_name, genes in gene_groups.items():
        enrichment_results[group_name] = {}
        for db in databases:
            logger.info(f"Analyzing {group_name} with database {db}...")
            try:
                df = get_enrichment_dataframe(genes, db, verbose=verbose)
                enrichment_results[group_name][db] = df
                logger.info(f"Analysis completed for {group_name} with {db}.")
            except Exception as e:
                logger.error(f"Error analyzing {group_name} with {db}: {e}")

    return enrichment_results


def subset_enrichment_results(
    enrichment_results: Dict[str, Dict[str, pd.DataFrame]],
    selected_database: str = 'GO_Biological_Process_2023',
    significance_threshold: float = 0.05,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Subset the enrichment results for a selected database and prepare a DataFrame for heatmap generation.

    Parameters
    ----------
    enrichment_results : Dict[str, Dict[str, pd.DataFrame]]
        Nested dictionary containing enrichment results from generate_enrichment_results.
    selected_database : str, optional
        The database to use for generating the heatmap. Default is 'GO_Biological_Process_2023'.
    significance_threshold : float, optional
        Threshold for significance in p-value. Default is 0.05.
    verbose : bool, optional
        If True, enable detailed logging. Default is True.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the merged enrichment results for the selected database.
        Rows represent terms, columns represent gene groups, and values are p-values.
    """
    set_logger_verbosity(verbose)
    logger.info(f"Subsetting enrichment results for database: {selected_database} with p-value threshold: {significance_threshold}")

    merged_df = pd.DataFrame()

    for group_name, db_results in enrichment_results.items():
        if selected_database in db_results:
            df = db_results[selected_database].copy()
            # Apply p-value threshold
            df['Significant'] = df['P-value'] <= significance_threshold
            # Retain p-values if significant, else set to NaN or a placeholder (e.g., 1 for non-significant)
            df['mlogP'] = -np.log10(df['P-value'])
            df['mlogP'] = df.apply(lambda row: row['mlogP'] if row['Significant'] else np.nan, axis=1)
            df = df[['Term', 'mlogP']]
            df = df.rename(columns={'mlogP': f'{group_name}'})
            if merged_df.empty:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on='Term', how='outer')

    # Replace NaN with a placeholder (e.g., the significance threshold or 1)
    merged_df = merged_df.fillna(0)  # Assuming 1 is non-significant

    # Set 'Term' as index
    merged_df.set_index('Term', inplace=True)

    logger.info(f"Merged DataFrame created with shape: {merged_df.shape}")

    return merged_df

def create_enrichment_clustermap(
    heatmap_df: pd.DataFrame,
    selected_database: str,
    term_pattern: Optional[str] = None,
    p_value_threshold: float = 0.05,
    cmap: str = "Reds",
    figsize: Tuple[int, int] = (14, 12),
    metric: str = "euclidean",
    method: str = "average",
    add_annotations: bool = True,
    verbose: bool = True
) -> sns.matrix.ClusterGrid:
    """
    Create a clustermap heatmap for enrichment results.

    Parameters
    ----------
    heatmap_df : pd.DataFrame
        DataFrame containing the merged enrichment results for the selected database.
        Rows represent terms, columns represent gene groups, and values are p-values.
    selected_database : str
        The database used for enrichment (used in the plot title).
    term_pattern : Optional[str], optional
        Regex pattern to filter terms. Only terms containing this pattern will be included.
        If None, no term filtering is applied. Default is None.
    p_value_threshold : float, optional
        Threshold for significance in p-value. Default is 0.05.
    cmap : str, optional
        Colormap for the heatmap. Default is "Reds".
    figsize : Tuple[int, int], optional
        Figure size for the heatmap. Default is (14, 12).
    metric : str, optional
        Distance metric for clustering. Default is "euclidean".
    method : str, optional
        Clustering linkage method. Default is "average".
    add_annotations : bool, optional
        Whether to add stars (*) to denote significant p-values. Default is True.
    verbose : bool, optional
        If True, enable detailed logging. Default is True.

    Returns
    -------
    sns.matrix.ClusterGrid
        The seaborn clustermap object.
    """
    set_logger_verbosity(verbose)
    logger.info("Starting clustermap generation.")

    # Step 1: Term Selection (if pattern provided)
    if term_pattern:
        logger.info(f"Filtering terms with pattern: '{term_pattern}' (case-insensitive)")
        heatmap_df = heatmap_df[heatmap_df.index.str.contains(term_pattern, case=False, na=False)]
        logger.debug(f"Number of terms after filtering: {heatmap_df.shape[0]}")

    # Step 2: P-value Filtering
    mlog10_threshold = -np.log10(p_value_threshold)
    logger.info(f"Applying p-value threshold: {mlog10_threshold}")
    # Keep terms where any p-value <= threshold
    heatmap_df_filtered = heatmap_df[(heatmap_df >= mlog10_threshold).any(axis=1)]
    logger.debug(f"Number of terms after p-value filtering: {heatmap_df_filtered.shape[0]}")

    if heatmap_df_filtered.empty:
        logger.warning("No terms meet the p-value threshold after filtering.")
        raise ValueError("No terms meet the p-value threshold after filtering.")

    logger.debug("Data transformation complete.")

    # Step 3: Generate Clustermap
    logger.info("Generating clustermap.")
    g = sns.clustermap(
        heatmap_df_filtered, 
        annot=False, 
        cmap=cmap, 
        figsize=figsize, 
        metric=metric, 
        method=method,
        standard_scale=None
    )
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=30, ha='right')

    # Step 4: Get reordered indices
    row_order = g.dendrogram_row.reordered_ind
    col_order = g.dendrogram_col.reordered_ind

    # Step 5: Adding Stars for Significant P-values
    if add_annotations:
        logger.info("Adding stars for significant p-values.")
        for i, term_idx in enumerate(row_order):
            for j, group_idx in enumerate(col_order):
                # Retrieve the original p-value
                term = heatmap_df_filtered.index[term_idx]
                group = heatmap_df_filtered.columns[group_idx]
                p_val = heatmap_df_filtered.loc[term, group]
                if p_val >= -np.log10(0.001):
                    symbol = '***'
                elif p_val >= -np.log10(0.01):
                    symbol = '**'
                elif p_val >= -np.log10(0.05):
                    symbol = '*'
                else:
                    symbol = ''
                if symbol:
                    g.ax_heatmap.text(
                        j + 0.5, 
                        i + 0.5, 
                        symbol, 
                        ha='center', 
                        va='center', 
                        color='black', 
                        fontsize=12
                    )

    # Step 6: Set Title
    plt.title(f"Enrichment Clustermap for {selected_database}", pad=20)

    logger.info("Clustermap generation complete.")
    plt.show()

    return g
