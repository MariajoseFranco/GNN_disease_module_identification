import pandas as pd


def mapping_diseases_to_proteins(df_dis_pro: pd.DataFrame) -> dict:
    """
    Map each disease to its associated proteins.

    Args:
        df_dis_pro (pd.DataFrame): DataFrame containing disease-protein associations.

    Returns:
        dict: Dictionary mapping disease names to sets of associated proteins.
    """
    disease_pro_mapping = {
        disease: dict(zip(group['protein_id'], group['score']))
        for disease, group in df_dis_pro.groupby("disease_name")
    }
    return disease_pro_mapping


def obtaining_predicted_proteins(node_index, preds, val_idx, seed_nodes):
    """
    Converts predicted test edges (indices) into protein-protein pairs.

    Args:
        node_index (dict): Mapping from node indices to protein names.
        preds (Tensor): Binary predictions for test edges.
        u_test (Tensor): Source node indices of test edges.
        v_test (Tensor): Destination node indices of test edges.

    Returns:
        list: List of predicted protein-protein interaction pairs (tuples).
    """
    index_node = {v: k for k, v in node_index.items()}

    # Get indices of predicted positives (label 1) among validation nodes
    predicted_positive_mask = preds == 1  # preds is of length len(val_idx)
    positive_indices = val_idx[predicted_positive_mask]

    # Get corresponding protein names
    predicted_positive_proteins = [index_node[int(idx)] for idx in positive_indices]
    df_predicted = pd.DataFrame(predicted_positive_proteins, columns=['Predicted Proteins'])
    df_predicted['is_seed'] = False
    for seed in seed_nodes:
        if seed in predicted_positive_proteins:
            df_predicted.loc[df_predicted['Predicted Proteins'] == seed, 'is_seed'] = True
    return df_predicted
