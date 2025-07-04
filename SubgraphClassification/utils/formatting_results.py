import json
import os

import pandas as pd
import torch


def obtaining_pred_scores(
    labels: torch.Tensor,
    preds: torch.Tensor,
    logits: torch.Tensor,
    test_idx: torch.Tensor,
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extracts the ground truth labels, predicted labels, and predicted scores for the test set.

    Args:
        labels (Tensor): Full label tensor.
        preds (Tensor): Binary predictions from the model.
        logits (Tensor): Raw output logits from the model.
        test_idx (Tensor): Indices of test nodes.
        device (torch.device): Computation device (CPU or GPU).

    Returns:
        tuple: (y_true, y_pred, y_scores) tensors for test set.
    """
    y_true = labels[test_idx].to(device)
    y_pred = preds.to(device)
    y_scores = logits[test_idx.to(device), 1].to(device)
    return y_true, y_pred, y_scores


def displaying_metrics(
    test_acc: float,
    test_f1: float,
    test_rec: float,
    test_prec: float,
    test_auc: float,
    preds: torch.Tensor,
    test_idx: torch.Tensor
) -> None:
    """
    Prints evaluation metrics and prediction statistics for the test set.

    Args:
        test_acc (float): Accuracy score on the test set.
        test_f1 (float): F1-score on the test set.
        test_rec (float): Recall score on the test set.
        test_prec (float): Precision score on the test set.
        test_auc (float): Area Under the ROC Curve (AUC) on the test set.
        preds (Tensor): Binary predictions for the test nodes.
        test_idx (Tensor): Indices corresponding to the test nodes.

    Returns:
        None
    """
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_rec:.4f}")
    print(f"Test AUC Score: {test_auc:.4f}")
    counts = torch.bincount(preds)
    if len(counts) == 1:
        counts = torch.cat((counts, torch.tensor([0])))
    print('\nTest set size: ', len(test_idx))
    print(f"Test predictions breakdown: Predicted {counts[0]} as 0 "
          f"(no seed nodes) and {counts[1]} as 1 (seed nodes)")


def saving_results(
    predicted_proteins: pd.DataFrame,
    seed_nodes: list[str],
    best_params: dict,
    y_true: torch.Tensor,
    y_scores: torch.Tensor,
    preds: torch.Tensor,
    model: torch.nn.Module,
    test_prec: float,
    test_rec: float,
    test_f1: float,
    test_auc: float,
    test_acc: float,
    best_threshold: float,
    labels: torch.Tensor,
    test_idx: torch.Tensor,
    node_index_filtered: dict[str, int],
    non_isolated_nodes: list[str],
    disease: str,
    output_path: str
) -> None:
    """
    Saves all relevant results for a trained model including predictions, scores, metrics,
    model weights, and configuration files for a given disease.

    Args:
        predicted_proteins (DataFrame): DataFrame of predicted proteins.
        seed_nodes (list): Ground-truth seed proteins.
        best_params (dict): Dictionary of best hyperparameters from Optuna.
        y_true, y_scores, preds (Tensor): Ground truth, probabilities, and predictions.
        model (nn.Module): Trained model.
        test_prec, test_rec, test_f1, test_auc, test_acc (float): Evaluation metrics.
        best_threshold (float): Chosen decision threshold.
        labels (Tensor): Full label vector.
        test_idx (Tensor): Indices for test set.
        node_index_filtered (dict): Mapping of protein names to indices.
        non_isolated_nodes (list): List of used protein nodes.
        disease (str): Disease name.
        output_path (str): Path to store the results.
    """
    saving_predicted(predicted_proteins, output_path, disease)
    saving_real_seeds(seed_nodes, output_path, disease)
    saving_hyperparams(best_params, output_path, disease)
    saving_results_tensors(
        y_true, y_scores, preds, model, disease, output_path
    )
    saving_metrics(
        test_prec, test_rec, test_f1, test_auc, test_acc, best_threshold,
        labels, preds, test_idx, output_path, disease
    )
    saving_others_info(node_index_filtered, non_isolated_nodes, output_path, disease)


def saving_predicted(predicted_proteins: pd.DataFrame, output_path: str, disease: str) -> None:
    """
    Saves the predicted proteins to a tab-separated file.

    Args:
        predicted_proteins (DataFrame): DataFrame of predicted proteins.
        output_path (str): Directory to save the file.
        disease (str): Disease name used for folder naming.
    """
    predicted_proteins.to_csv(
        f"{output_path}/{disease}"
        f"/predicted_proteins.txt", sep="\t", index=False
    )


def saving_real_seeds(seed_nodes: list[str], output_path: str, disease: str) -> None:
    """
    Writes the list of real seed proteins to a text file.

    Args:
        seed_nodes (list): List of known proteins.
        output_path (str): Output directory.
        disease (str): Disease name for directory naming.
    """
    with open(
        f"{output_path}/{disease}/real_seeds_proteins.txt", "w"
    ) as f:
        for seed in seed_nodes:
            f.write(f"{seed}\n")


def saving_hyperparams(best_params: dict, output_path: str, disease: str) -> None:
    """
    Saves the best hyperparameters as a JSON file.

    Args:
        best_params (dict): Dictionary of optimal hyperparameters.
        output_path (str): Output directory path.
        disease (str): Disease name for subfolder.
    """
    os.makedirs(f'{output_path}/{disease}/model', exist_ok=True)
    with open(f"{output_path}/{disease}/model/best_hyperparams.json", "w") as f:
        json.dump(best_params, f, indent=4)


def saving_results_tensors(
    y_true: torch.Tensor,
    y_scores: torch.Tensor,
    preds: torch.Tensor,
    model: torch.nn.Module,
    disease: str,
    output_path: str
) -> None:
    """
    Saves the evaluation results (true labels, predicted scores, predictions) and model weights.

    Args:
        y_true (Tensor): Ground truth labels.
        y_scores (Tensor): Predicted probabilities.
        preds (Tensor): Binary predictions.
        model (nn.Module): Trained GNN model.
        disease (str): Disease identifier.
        output_path (str): Directory to save the files.
    """
    torch.save(y_scores.cpu(), f"{output_path}/{disease}/model/y_scores.pt")
    torch.save(y_true.cpu(), f"{output_path}/{disease}/model/y_true.pt")
    torch.save(preds.cpu(), f"{output_path}/{disease}/model/preds.pt")
    torch.save(model.state_dict(), f"{output_path}/{disease}/model/model.pt")


def obtaining_metrics_dict(
    test_prec: float,
    test_rec: float,
    test_f1: float,
    test_auc: float,
    test_acc: float,
    best_threshold: float,
    labels: torch.Tensor,
    preds: torch.Tensor,
    test_idx: torch.Tensor
) -> dict:
    """
    Creates a dictionary with all relevant evaluation metrics and metadata.

    Args:
        test_prec, test_rec, test_f1, test_auc, test_acc (float): Evaluation scores.
        best_threshold (float): Chosen threshold for classification.
        labels (Tensor): All labels.
        preds (Tensor): Predictions over all data.
        test_idx (Tensor): Indices for test set.

    Returns:
        dict: Dictionary of evaluation metrics and statistics.
    """
    metrics_dict = {
        "test_precision": float(test_prec),
        "test_recall": float(test_rec),
        "test_f1": float(test_f1),
        "test_auc": float(test_auc),
        "test_accuracy": float(test_acc),
        "threshold": float(best_threshold),
        "true_positives": int((labels[test_idx] == 1).sum().item()),
        "true_negatives": int((labels[test_idx] == 0).sum().item()),
        "predicted_positives": int((preds == 1).sum().item()),
        "predicted_negatives": int((preds == 0).sum().item()),
        "num_total_predictions": int(preds.shape[0])
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
    preds: torch.Tensor,
    test_idx: torch.Tensor,
    output_path: str,
    disease: str
) -> None:
    """
    Writes the evaluation metrics to a JSON file.

    Args:
        All standard metrics and identifiers needed to compute the result summary.
    """
    metrics_dict = obtaining_metrics_dict(
        test_prec, test_rec, test_f1, test_auc, test_acc, best_threshold, labels, preds, test_idx
    )

    with open(f"{output_path}/{disease}/model/evaluation_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)


def saving_others_info(
    node_index_filtered: dict[str, int],
    non_isolated_nodes: list[str],
    output_path: str,
    disease: str
) -> None:
    """
    Saves additional metadata like filtered node index and used nodes.

    Args:
        node_index_filtered (dict): Protein-to-index mapping.
        non_isolated_nodes (list): List of protein names used in training.
        output_path (str): Output folder path.
        disease (str): Disease name.
    """
    with open(f"{output_path}/{disease}/model/node_index.json", "w") as f:
        json.dump(node_index_filtered, f, indent=4)

    with open(f"{output_path}/{disease}/model/nodes_used.json", "w") as f:
        json.dump(non_isolated_nodes, f, indent=4)
