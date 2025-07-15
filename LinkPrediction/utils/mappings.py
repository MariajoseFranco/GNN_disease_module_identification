from typing import Dict, Tuple

import pandas as pd


def mapping_dis_pro_edges_to_scores(df_dis_pro: pd.DataFrame) -> Dict[Tuple[int, int], float]:
    """
    Maps disease-protein edges to their corresponding association scores.

    Args:
        df_dis_pro (pd.DataFrame): DataFrame containing disease-protein associations.
            Required columns:
                - 'disease_id': Encoded disease node ID.
                - 'protein_id_enc': Encoded protein node ID.
                - 'score': Association score between the disease and protein.

    Returns:
        Dict[Tuple[int, int], float]: Dictionary mapping (disease_id, protein_id_enc) pairs
        to their corresponding scores.
    """
    return {
        (row['disease_id'], row['protein_id_enc']): row['score']
        for _, row in df_dis_pro.iterrows()
    }


def mapping_dis_pro_edges_to_scores_full_graph(
        df_dis_pro: pd.DataFrame
) -> Dict[Tuple[int, int], float]:
    """
    Maps disease-protein edges to association scores in the full heterogeneous graph setting.

    Args:
        df_dis_pro (pd.DataFrame): DataFrame containing disease-protein associations
        for the full graph.
            Required columns:
                - 'dis': Encoded disease node ID.
                - 'pro': Encoded protein node ID.
                - 'w': Weight or association score.

    Returns:
        Dict[Tuple[int, int], float]: Dictionary mapping (dis, pro) pairs to their
        corresponding weights.
    """
    return {
        (row['dis'], row['pro']): row['w']
        for _, row in df_dis_pro.iterrows()
    }


def mapping_to_encoded_ids(
    df_dis_pro: pd.DataFrame,
    df_pro_pro: pd.DataFrame
) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Create dictionaries mapping node indices (encoded IDs) to original node names
    for diseases and proteins.

    Args:
        df_dis_pro (pd.DataFrame): DataFrame containing disease-protein associations.
            Required columns for disease mapping:
                - 'disease_id_enc': Encoded disease node IDs.
                - 'disease_name': Original disease names.
            Required columns for protein mapping (partial input).

        df_pro_pro (pd.DataFrame): DataFrame containing protein-protein interactions.
            Required for protein mapping:
                - 'protein_id_enc': Encoded protein node IDs.
                - 'protein_id': Original protein identifiers.

    Returns:
        Tuple[Dict[int, str], Dict[int, str]]:
            - diseases_to_encoded_ids: Mapping from encoded disease node IDs to disease names.
            - proteins_to_encoded_ids: Mapping from encoded protein node IDs to protein names.
    """
    diseases_to_encoded_ids = mapping_diseases_to_encoded_ids(df_dis_pro)
    proteins_to_encoded_ids = mapping_proteins_to_encoded_ids(df_dis_pro, df_pro_pro)
    return diseases_to_encoded_ids, proteins_to_encoded_ids


def mapping_diseases_to_encoded_ids(df_dis_pro: pd.DataFrame) -> Dict[int, str]:
    """
    Creates a mapping from encoded disease IDs to disease names.

    Args:
        df_dis_pro (pd.DataFrame): DataFrame containing disease-protein associations.
            Required columns:
                - 'disease_id': Encoded disease ID.
                - 'disease_name': Original disease name.

    Returns:
        Dict[int, str]: Dictionary mapping encoded disease IDs to disease names.
    """
    diseases_to_encoded_ids = {
        idx: disease for idx, disease in df_dis_pro[['disease_id', 'disease_name']]
        .drop_duplicates()
        .values
    }
    return diseases_to_encoded_ids


def mapping_proteins_to_encoded_ids(
    df_dis_pro: pd.DataFrame,
    df_pro_pro: pd.DataFrame
) -> Dict[int, str]:
    """
    Creates a mapping from encoded protein IDs to protein names by combining information
    from disease-protein associations and protein-protein interactions.

    Args:
        df_dis_pro (pd.DataFrame): DataFrame containing disease-protein associations.
            Required columns:
                - 'protein_id_enc': Encoded protein ID.
                - 'protein_id': Original protein identifier.

        df_pro_pro (pd.DataFrame): DataFrame containing protein-protein interactions.
            Required columns:
                - 'src_id': Encoded protein ID (source node).
                - 'prA': Original protein identifier (source).
                - 'dst_id': Encoded protein ID (destination node).
                - 'prB': Original protein identifier (destination).

    Returns:
        Dict[int, str]: Dictionary mapping encoded protein IDs to protein identifiers.
    """
    # Extract mappings from disease-protein associations
    dis_pro_mapping = df_dis_pro[['protein_id_enc', 'protein_id']].drop_duplicates()

    # Extract mappings from protein-protein interactions
    src_mapping = df_pro_pro[['src_id', 'prA']].drop_duplicates()
    dst_mapping = df_pro_pro[['dst_id', 'prB']].drop_duplicates()

    # Standardize column names for concatenation
    src_mapping.columns = ['protein_id_enc', 'protein_id']
    dst_mapping.columns = ['protein_id_enc', 'protein_id']

    # Combine all mappings and remove duplicates
    combined_mapping = pd.concat([dis_pro_mapping, src_mapping, dst_mapping]).drop_duplicates()

    # Create final dictionary
    proteins_to_encoded_ids = {
        idx: protein for idx, protein in combined_mapping.values
    }
    return proteins_to_encoded_ids
