from typing import Dict, Set

import pandas as pd
import torch


def mapping_diseases_to_proteins(df_dis_pro: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Maps each disease to its associated proteins with corresponding scores.

    Args:
        df_dis_pro (pd.DataFrame): DataFrame containing columns 'disease_name',
                                   'protein_id', and 'score'.

    Returns:
        dict: Dictionary where keys are disease names and values are dictionaries
              mapping protein IDs to their association scores.
    """
    disease_pro_mapping = {
        disease: dict(zip(group['protein_id'], group['score']))
        for disease, group in df_dis_pro.groupby("disease_name")
    }
    return disease_pro_mapping


def obtaining_predicted_proteins(
    node_index: Dict[str, int],
    preds: torch.Tensor,
    val_idx: torch.Tensor,
    seed_nodes: Set[str]
) -> pd.DataFrame:
    """
    Maps predicted node indices back to protein names and identifies which ones are seeds.

    Args:
        node_index (dict): Mapping from protein names to node indices.
        preds (torch.Tensor): Binary predictions for validation/test set (0 or 1).
        val_idx (torch.Tensor): Indices corresponding to the validation/test set.
        seed_nodes (set): Set of known disease-associated proteins (positives).

    Returns:
        pd.DataFrame: DataFrame with two columns:
            - 'Predicted Proteins': protein names predicted as positive
            - 'is_seed': boolean indicating if the predicted protein is a known seed
    """
    index_node = {v: k for k, v in node_index.items()}

    predicted_positive_mask = preds == 1
    positive_indices = val_idx[predicted_positive_mask]

    predicted_positive_proteins = [index_node[int(idx)] for idx in positive_indices]
    df_predicted = pd.DataFrame(predicted_positive_proteins, columns=['Predicted Proteins'])
    df_predicted['is_seed'] = False
    for seed in seed_nodes:
        if seed in predicted_positive_proteins:
            df_predicted.loc[df_predicted['Predicted Proteins'] == seed, 'is_seed'] = True
    return df_predicted
