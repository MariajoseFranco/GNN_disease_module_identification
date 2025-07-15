import json
import os
import pickle
from typing import Dict, Tuple

import pandas as pd
import torch
from torch import nn


def obtaining_dis_pro_predicted(
    preds: torch.Tensor,
    test_pos_u: torch.Tensor,
    test_pos_v: torch.Tensor,
    test_neg_u: torch.Tensor,
    test_neg_v: torch.Tensor,
    diseases_to_encoded_ids: Dict[int, str],
    proteins_to_encoded_ids: Dict[int, str],
    seed_nodes: Dict[str, Tuple[str]]
) -> pd.DataFrame:
    """
    Maps predicted positive edges to disease-protein pairs and annotates each prediction
    with whether it corresponds to a known seed protein for that disease.

    Args:
        preds (torch.Tensor): Binary predictions (1 for predicted positive
        edges, 0 otherwise).
        test_pos_u (torch.Tensor): Source node indices for positive test
        edges (disease nodes).
        test_pos_v (torch.Tensor): Destination node indices for positive
        test edges (protein nodes).
        test_neg_u (torch.Tensor): Source node indices for negative test edges (disease nodes).
        test_neg_v (torch.Tensor): Destination node indices for negative
        test edges (protein nodes).
        diseases_to_encoded_ids (Dict[int, str]): Mapping from encoded disease node
        indices to disease names.
        proteins_to_encoded_ids (Dict[int, str]): Mapping from encoded protein node
        indices to protein names.
        seed_nodes (Dict[str, Tuple[str]]): Mapping from disease names to tuples of
        their known seed proteins.

    Returns:
        pd.DataFrame: DataFrame with columns ['disease', 'protein', 'is_seed_node'],
                    containing the predicted disease-protein associations.
    """
    # Concatenate positive and negative test edges
    all_u = torch.cat([test_pos_u, test_neg_u])
    all_v = torch.cat([test_pos_v, test_neg_v])

    # Select predicted positive edges (preds == 1)
    predicted_indices = (preds == 1).nonzero(as_tuple=True)[0]
    predicted_edges = [(all_u[i].item(), all_v[i].item()) for i in predicted_indices]

    # Map indices to disease and protein names
    predicted_named_edges = [
        (diseases_to_encoded_ids[u], proteins_to_encoded_ids[v])
        for u, v in predicted_edges
    ]

    predicted_df = pd.DataFrame(predicted_named_edges, columns=['disease', 'protein'])
    predicted_df['is_seed_node'] = False

    # Annotate whether each predicted edge corresponds to a known seed protein
    for disease, seeds in seed_nodes.items():
        mask = predicted_df['disease'] == disease
        predicted_df.loc[mask, 'is_seed_node'] = predicted_df.loc[mask, 'protein'].isin(seeds)

    # Sort by disease for readability
    predicted_df = predicted_df.sort_values(by='disease').reset_index(drop=True)

    return predicted_df


def obtaining_dis_dru_predicted(
    pos_score: torch.Tensor,
    neg_score: torch.Tensor,
    test_pos_u: torch.Tensor,
    test_pos_v: torch.Tensor,
    test_neg_u: torch.Tensor,
    test_neg_v: torch.Tensor,
    diseases_to_encoded_ids: Dict[int, str],
    drugs_to_encoded_ids: Dict[int, str]
) -> pd.DataFrame:
    """
    Maps predicted disease-drug pairs from model outputs to human-readable names
    and annotates whether each predicted pair corresponds to a known association.

    The function concatenates positive and negative test predictions,
    applies a sigmoid to convert logits to probabilities,
    applies a 0.5 threshold to select predicted positive edges,
    and then maps the node indices to disease and drug names.

    Additionally, it marks whether each predicted edge is a
    true positive (known association) or a novel prediction.

    Args:
        pos_score (torch.Tensor): Model scores for positive test edges.
        neg_score (torch.Tensor): Model scores for negative test edges.
        test_pos_u (torch.Tensor): Source node indices (diseases) of positive test edges.
        test_pos_v (torch.Tensor): Target node indices (drugs) of positive test edges.
        test_neg_u (torch.Tensor): Source node indices (diseases) of negative test edges.
        test_neg_v (torch.Tensor): Target node indices (drugs) of negative test edges.
        diseases_to_encoded_ids (Dict[int, str]): Mapping from disease node
        indices to disease names.
        drugs_to_encoded_ids (Dict[int, str]): Mapping from drug node indices to drug names.

    Returns:
        pd.DataFrame: A DataFrame with columns:
            - 'disease': Disease name.
            - 'drug': Drug name.
            - 'known_association': Boolean indicating if the predicted pair is in the
            test set positive edges.
    """
    all_scores = torch.cat([pos_score, neg_score])
    probs = torch.sigmoid(all_scores)
    preds = (probs >= 0.5).int()

    # Concatenate test edges
    all_u = torch.cat([test_pos_u, test_neg_u])
    all_v = torch.cat([test_pos_v, test_neg_v])

    # Set of known associations from test_pos edges
    known_associations_set = set(zip(
        test_pos_u.tolist(),
        test_pos_v.tolist()
    ))

    # Select predicted positive edges
    predicted_indices = (preds == 1).nonzero(as_tuple=True)[0]
    predicted_edges = [(all_u[i].item(), all_v[i].item()) for i in predicted_indices]

    # Map node IDs to names and annotate known associations
    predicted_named_edges = []
    known_flags = []
    for u, v in predicted_edges:
        dis = diseases_to_encoded_ids[u]
        dru = drugs_to_encoded_ids[v]
        predicted_named_edges.append((dis, dru))
        known_flags.append((u, v) in known_associations_set)

    predicted_df = pd.DataFrame(predicted_named_edges, columns=['disease', 'drug'])
    predicted_df['known_association'] = known_flags
    predicted_df = predicted_df.sort_values(by='disease').reset_index(drop=True)
    return predicted_df


def displaying_metrics(
    test_acc: float,
    test_f1: float,
    test_rec: float,
    test_prec: float,
    test_auc: float,
    test_preds: torch.Tensor
) -> None:
    print(f"\nTest Accuracy: {test_acc:.4f} | Test AUC: {test_auc:.4f} | "
          f"Test Precision: {test_prec:.4f} | Test Recall: {test_rec:.4f} | "
          f"Test F1: {test_f1:.4f}")

    counts = torch.bincount(test_preds)
    if len(counts) == 1:
        counts = torch.cat((counts, torch.tensor([0])))
    print('\nTest preds set size: ', len(test_preds))
    print(f"Test predictions breakdown: Predicted {counts[0]} as 0 "
          f"(no seed nodes) and {counts[1]} as 1 (seed nodes)")


def saving_results(
    predicted_dis_pro: pd.DataFrame,
    best_params: dict,
    best_threshold: float,
    test_probs: torch.Tensor,
    test_preds: torch.Tensor,
    labels: torch.Tensor,
    model: nn.Module,
    test_pos_u: torch.Tensor,
    test_pos_v: torch.Tensor,
    test_neg_u: torch.Tensor,
    test_neg_v: torch.Tensor,
    test_prec: float,
    test_rec: float,
    test_f1: float,
    test_auc: float,
    test_acc: float,
    diseases_to_encoded_ids: Dict[int, str],
    proteins_to_encoded_ids: Dict[int, str],
    output_path: str
) -> None:
    """
    Saves all relevant outputs from the link prediction pipeline, including predictions,
    model parameters, evaluation metrics, and additional mappings.

    Args:
        predicted_dis_pro (pd.DataFrame): Predicted disease-protein associations with annotations.
        best_params (dict): Best hyperparameters found during tuning.
        best_threshold (float): Best classification threshold found during validation.
        test_probs (torch.Tensor): Predicted probabilities from the model.
        test_preds (torch.Tensor): Binary predictions from the model.
        labels (torch.Tensor): Ground truth labels.
        model (nn.Module): Trained model.
        test_pos_u, test_pos_v (torch.Tensor): Positive test edge sources and destinations.
        test_neg_u, test_neg_v (torch.Tensor): Negative test edge sources and destinations.
        test_prec, test_rec, test_f1, test_auc, test_acc (float): Evaluation metrics.
        diseases_to_encoded_ids (dict): Mapping from encoded IDs to disease names.
        proteins_to_encoded_ids (dict): Mapping from encoded IDs to protein names.
        output_path (str): Directory where outputs will be saved.
    """
    saving_predicted(predicted_dis_pro, output_path)
    saving_hyperparams(best_params, best_threshold, output_path)
    saving_results_tensors(
        test_probs, test_preds, labels, model, test_pos_u,
        test_pos_v, test_neg_u, test_neg_v, output_path
    )
    saving_metrics(
        test_prec, test_rec, test_f1, test_auc, test_acc,
        best_threshold, labels, test_preds, output_path
    )
    saving_others_info(diseases_to_encoded_ids, proteins_to_encoded_ids, output_path)


def saving_results_full_graph(
    predicted_dis_pro: pd.DataFrame,
    best_params: dict,
    best_threshold: float,
    test_probs: torch.Tensor,
    test_preds: torch.Tensor,
    labels: torch.Tensor,
    model: nn.Module,
    test_pos_u: torch.Tensor,
    test_pos_v: torch.Tensor,
    test_neg_u: torch.Tensor,
    test_neg_v: torch.Tensor,
    test_prec: float,
    test_rec: float,
    test_f1: float,
    test_auc: float,
    test_acc: float,
    diseases_to_encoded_ids: Dict[int, str],
    drugs_to_encoded_ids: Dict[int, str],
    output_path: str
) -> None:
    """
    Saves all relevant outputs from the link prediction pipeline, including predictions,
    model parameters, evaluation metrics, and additional mappings.

    Args:
        predicted_dis_pro (pd.DataFrame): Predicted disease-protein associations with annotations.
        best_params (dict): Best hyperparameters found during tuning.
        best_threshold (float): Best classification threshold found during validation.
        test_probs (torch.Tensor): Predicted probabilities from the model.
        test_preds (torch.Tensor): Binary predictions from the model.
        labels (torch.Tensor): Ground truth labels.
        model (nn.Module): Trained model.
        test_pos_u, test_pos_v (torch.Tensor): Positive test edge sources and destinations.
        test_neg_u, test_neg_v (torch.Tensor): Negative test edge sources and destinations.
        test_prec, test_rec, test_f1, test_auc, test_acc (float): Evaluation metrics.
        diseases_to_encoded_ids (dict): Mapping from encoded IDs to disease names.
        drugs_to_encoded_ids (dict): Mapping from encoded IDs to protein names.
        output_path (str): Directory where outputs will be saved.
    """
    saving_predicted_dis_dru(predicted_dis_pro, output_path)
    saving_hyperparams(best_params, best_threshold, output_path)
    saving_results_tensors(
        test_probs, test_preds, labels, model, test_pos_u,
        test_pos_v, test_neg_u, test_neg_v, output_path
    )
    saving_metrics(
        test_prec, test_rec, test_f1, test_auc, test_acc,
        best_threshold, labels, test_preds, output_path
    )
    saving_others_info_full_graph(diseases_to_encoded_ids, drugs_to_encoded_ids, output_path)


def saving_predicted(predicted_dis_pro: pd.DataFrame, output_path: str) -> None:
    """
    Save the predicted disease-protein associations to a TSV file.

    Args:
        predicted_dis_pro (pd.DataFrame): DataFrame containing predicted associations.
        output_path (str): Path to the directory where the file will be saved.
    """
    predicted_dis_pro.to_csv(
        f"{output_path}/predicted_dis_pro.txt",
        sep="\t",
        index=False
    )


def saving_predicted_dis_dru(predicted_dis_dru: pd.DataFrame, output_path: str) -> None:
    """
    Save the predicted disease-drugs associations to a TSV file.

    Args:
        predicted_dis_dru (pd.DataFrame): DataFrame containing predicted associations.
        output_path (str): Path to the directory where the file will be saved.
    """
    predicted_dis_dru.to_csv(
        f"{output_path}/predicted_dis_dru.txt",
        sep="\t",
        index=False
    )


def saving_hyperparams(best_params: dict, best_threshold: float, output_path: str) -> None:
    """
    Save the best hyperparameters and the optimal threshold to files.

    Args:
        best_params (dict): Dictionary of selected hyperparameters from tuning.
        best_threshold (float): Optimal threshold for classification.
        output_path (str): Path to the directory where files will be saved.
    """
    os.makedirs(f'{output_path}/model', exist_ok=True)
    with open(f"{output_path}/model/best_hyperparams.json", "w") as f:
        json.dump(best_params, f, indent=4)
    with open(f"{output_path}/model/best_threshold.txt", "w") as f:
        f.write(str(best_threshold))


def saving_results_tensors(
    test_probs: torch.Tensor,
    test_preds: torch.Tensor,
    labels: torch.Tensor,
    model: nn.Module,
    test_pos_u: torch.Tensor,
    test_pos_v: torch.Tensor,
    test_neg_u: torch.Tensor,
    test_neg_v: torch.Tensor,
    output_path: str
) -> None:
    """
    Save model outputs, predictions, and test edges to disk.

    Args:
        test_probs (torch.Tensor): Predicted probabilities.
        test_preds (torch.Tensor): Predicted labels (0/1).
        labels (torch.Tensor): Ground truth labels.
        model (nn.Module): Trained model.
        test_pos_u (torch.Tensor): Positive edge sources in test set.
        test_pos_v (torch.Tensor): Positive edge destinations in test set.
        test_neg_u (torch.Tensor): Negative edge sources in test set.
        test_neg_v (torch.Tensor): Negative edge destinations in test set.
        output_path (str): Path to save the outputs.
    """
    torch.save(test_probs.cpu(), f"{output_path}/model/y_scores.pt")
    torch.save(labels.cpu(), f"{output_path}/model/y_true.pt")
    torch.save(test_preds.cpu(), f"{output_path}/model/preds.pt")
    torch.save(model.state_dict(), f"{output_path}/model/model.pt")

    torch.save(test_pos_u, f"{output_path}/model/test_pos_u.pt")
    torch.save(test_pos_v, f"{output_path}/model/test_pos_v.pt")
    torch.save(test_neg_u, f"{output_path}/model/test_neg_u.pt")
    torch.save(test_neg_v, f"{output_path}/model/test_neg_v.pt")


def obtaining_metrics_dict(
    test_prec: float,
    test_rec: float,
    test_f1: float,
    test_auc: float,
    test_acc: float,
    best_threshold: float,
    labels: torch.Tensor,
    test_preds: torch.Tensor
) -> dict:
    """
    Collects evaluation metrics and summary statistics into a dictionary.

    Args:
        test_prec (float): Precision score.
        test_rec (float): Recall score.
        test_f1 (float): F1 score.
        test_auc (float): AUC-ROC score.
        test_acc (float): Accuracy.
        best_threshold (float): Best decision threshold.
        labels (torch.Tensor): Ground truth labels.
        test_preds (torch.Tensor): Predicted labels.

    Returns:
        dict: Dictionary containing metrics and prediction counts.
    """
    metrics_dict = {
        "test_precision": float(test_prec),
        "test_recall": float(test_rec),
        "test_f1": float(test_f1),
        "test_auc": float(test_auc),
        "test_accuracy": float(test_acc),
        "threshold": float(best_threshold),
        "true_positives": int((labels == 1).sum().item()),
        "true_negatives": int((labels == 0).sum().item()),
        "predicted_positives": int((test_preds == 1).sum().item()),
        "predicted_negatives": int((test_preds == 0).sum().item()),
        "num_total_predictions": int(test_preds.shape[0])
    }
    return metrics_dict


def saving_metrics(
    test_prec: float,
    test_rec: float,
    test_f1: float,
    test_auc: float,
    test_acc: float,
    best_threshold: float,
    labels: torch.Tensor,
    test_preds: torch.Tensor,
    output_path: str
) -> None:
    """
    Save evaluation metrics to a JSON file.

    Args:
        test_prec (float): Precision score.
        test_rec (float): Recall score.
        test_f1 (float): F1 score.
        test_auc (float): AUC-ROC score.
        test_acc (float): Accuracy.
        best_threshold (float): Best classification threshold.
        labels (torch.Tensor): Ground truth labels.
        test_preds (torch.Tensor): Predicted labels.
        output_path (str): Directory to save the metrics file.
    """
    metrics_dict = obtaining_metrics_dict(
        test_prec, test_rec, test_f1, test_auc, test_acc, best_threshold, labels, test_preds
    )

    metrics_path = f"{output_path}/model/evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)


def saving_others_info(
    diseases_to_encoded_ids: Dict[int, str],
    proteins_to_encoded_ids: Dict[int, str],
    output_path: str
) -> None:
    """
    Save mapping dictionaries (IDs to names) for diseases and proteins.

    Args:
        diseases_to_encoded_ids (dict): Mapping from encoded disease IDs to names.
        proteins_to_encoded_ids (dict): Mapping from encoded protein IDs to names.
        output_path (str): Directory where pickles will be saved.
    """
    with open(f"{output_path}/model/diseases_to_encoded_ids.pkl", "wb") as f:
        pickle.dump(diseases_to_encoded_ids, f)

    with open(f"{output_path}/model/proteins_to_encoded_ids.pkl", "wb") as f:
        pickle.dump(proteins_to_encoded_ids, f)


def saving_others_info_full_graph(
    diseases_to_encoded_ids: Dict[int, str],
    drugs_to_encoded_ids: Dict[int, str],
    output_path: str
) -> None:
    """
    Save mapping dictionaries (IDs to names) for diseases and proteins.

    Args:
        diseases_to_encoded_ids (dict): Mapping from encoded disease IDs to names.
        drugs_to_encoded_ids (dict): Mapping from encoded drugs IDs to names.
        output_path (str): Directory where pickles will be saved.
    """
    with open(f"{output_path}/model/diseases_to_encoded_ids.pkl", "wb") as f:
        pickle.dump(diseases_to_encoded_ids, f)

    with open(f"{output_path}/model/drugs_to_encoded_ids.pkl", "wb") as f:
        pickle.dump(drugs_to_encoded_ids, f)
