import json
import requests
import numpy as np
import pandas as pd
import io
import logging
from typing import List, Dict

# Configure logging for the enrichment module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SPIX.enrichment')


def get_gene_array(filename: str) -> List[str]:
    """
    Load a list of genes from a text file.

    Parameters
    ----------
    filename : str
        Path to the gene list file.

    Returns
    -------
    List[str]
        List of gene symbols.
    """
    try:
        genes = np.loadtxt(filename, dtype='object', unpack=True)
        genes = genes.tolist()
        logger.info(f"Loaded {len(genes)} genes from {filename}.")
        return genes
    except Exception as e:
        logger.error(f"Error loading genes from {filename}: {e}")
        raise


def get_background_array(filename: str) -> List[str]:
    """
    Load a list of background genes from a text file.

    Parameters
    ----------
    filename : str
        Path to the background gene list file.

    Returns
    -------
    List[str]
        List of background gene symbols.
    """
    try:
        background = np.loadtxt(filename, dtype='object', unpack=True)
        background = background.tolist()
        logger.info(f"Loaded {len(background)} background genes from {filename}.")
        return background
    except Exception as e:
        logger.error(f"Error loading background genes from {filename}: {e}")
        raise


def write_enrichment_file(genes: List[str], database: str, export_filename: str) -> None:
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
    """
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
    logger.info(f"Enrichr Add List Response: {data}")

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
    export_filename: str
) -> pd.DataFrame:
    """
    Perform background enrichment analysis using SpeedRICHr and return the results as a DataFrame.

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

    Returns
    -------
    pd.DataFrame
        DataFrame containing the enrichment results.
    """
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
    logger.info(f"User List Response: {userlist_response}")

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
    logger.info(f"Background Response: {background_response}")

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


def get_enrichment_dataframe(genes: List[str], database: str) -> pd.DataFrame:
    """
    Analyze a gene set using Enrichr and return the enrichment results as a pandas DataFrame.

    Parameters
    ----------
    genes : List[str]
        List of gene symbols.
    database : str
        Name of the Enrichr database to use (e.g., "KEGG_2021_Human").

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
    logger.info(f"Enrichr Add List Response: {data}")

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
    logger.info(f"Enrichment results loaded into DataFrame with {df.shape[0]} terms.")

    return df


def write_background_enrichment_df(
    genes: List[str],
    background: List[str],
    database: str,
    export_filename: str
) -> pd.DataFrame:
    """
    Perform background enrichment analysis using SpeedRICHr and return the results as a DataFrame.

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

    Returns
    -------
    pd.DataFrame
        DataFrame containing the enrichment results.
    """
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
    logger.info(f"User List Response: {userlist_response}")

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
    logger.info(f"Background Response: {background_response}")

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


def analyze_gene_sets(
    gene_groups: Dict[str, List[str]],
    databases: List[str],
    selected_database: str = 'GO_Biological_Process_2023',
    significance_threshold: float = 1.3
) -> pd.DataFrame:
    """
    Analyze multiple gene groups across multiple databases and generate a clustermap heatmap.

    Parameters
    ----------
    gene_groups : Dict[str, List[str]]
        Dictionary where keys are group names and values are lists of gene symbols.
    databases : List[str]
        List of database names to perform enrichment analysis.
    selected_database : str, optional
        The database to use for generating the heatmap. Default is 'GO_Biological_Process_2023'.
    significance_threshold : float, optional
        Threshold for significance in -log10(p-value). Default is 1.3 (~ p=0.05).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the merged enrichment results for the selected database.
    """
    # Store enrichment results
    enrichment_results = {}

    # Perform enrichment analysis
    for group_name, genes in gene_groups.items():
        enrichment_results[group_name] = {}
        for db in databases:
            logger.info(f"Analyzing {group_name} with database {db}...")
            try:
                df = get_enrichment_dataframe(genes, db)
                enrichment_results[group_name][db] = df
                logger.info(f"Analysis completed for {group_name} with {db}.")
            except Exception as e:
                logger.error(f"Error analyzing {group_name} with {db}: {e}")

    # Prepare data for heatmap
    merged_df = pd.DataFrame()

    for group_name in gene_groups.keys():
        if selected_database in enrichment_results[group_name]:
            df = enrichment_results[group_name][selected_database].copy()
            df['mlogP'] = -np.log10(df['P-value'])
            df = df[['Term', 'mlogP']]
            df = df.rename(columns={'mlogP': f'{group_name}'})
            if merged_df.empty:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on='Term', how='outer')

    # Replace NaN with 0
    merged_df = merged_df.fillna(0)

    # Set 'Term' as index
    merged_df.set_index('Term', inplace=True)

    # Filter terms containing 'Develop' (case-insensitive)
    p_2 = merged_df[merged_df.index.str.contains('Develop', case=False, na=False)]

    # Apply significance threshold
    p_2 = p_2[(p_2 >= significance_threshold).any(axis=1)]

    # Generate clustermap
    import seaborn as sns
    import matplotlib.pyplot as plt

    logger.info(f"Generating clustermap for database: {selected_database}")

    g = sns.clustermap(
        p_2,
        annot=False,
        cmap="Reds",
        figsize=(14, 12),
        standard_scale=None,
        metric="euclidean",
        method="average"
    )
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=30)

    # Retrieve the order of rows and columns
    row_order = g.dendrogram_row.reordered_ind
    col_order = g.dendrogram_col.reordered_ind

    # Add stars for significant terms
    for i, term_idx in enumerate(row_order):
        for j, group_idx in enumerate(col_order):
            if p_2.iloc[term_idx, group_idx] >= significance_threshold:
                g.ax_heatmap.text(
                    j + 0.5,
                    i + 0.7,
                    '*',
                    ha='center',
                    va='center',
                    color='black',
                    fontsize=12
                )

    # Set title
    plt.title(f"Enrichment Clustermap for {selected_database}")
    plt.show()

    return merged_df