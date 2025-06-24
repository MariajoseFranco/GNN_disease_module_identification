import os
import warnings

import pandas as pd
import torch
from focal_loss import FocalLoss
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn

from data_compilation import DataCompilation
from SubgraphClassification.GNN_sage import GNN
from SubgraphClassification.homogeneous_graph import HomogeneousGraph
from utils import (generate_expanded_labels, load_config,
                   mapping_diseases_to_proteins, split_train_test_val_indices)
from visualizations import (plot_confusion_matrix, plot_loss_and_metrics,
                            plot_precision_recall_curve)

warnings.filterwarnings("ignore")


class Main():
    def __init__(self):
        # Paths
        self.config = load_config()
        self.data_path = self.config['data_dir']
        self.disease_path = self.config['disease_dir']
        self.output_path = self.config['results_subg_dir']
        os.makedirs(f'{self.output_path}', exist_ok=True)

        self.DC = DataCompilation(self.data_path, self.disease_path, self.output_path)
        self.HomoGraph = HomogeneousGraph()
        self.epochs = 200
        self.best_threshold = -1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def training_loop(self, model, g, labels, train_idx, val_idx, optimizer, loss_fn):
        """
        Trains the GNN model for link prediction using positive and negative edges.

        Args:
            model (nn.Module): The GNN model.
            predictor (nn.Module): The predictor module that computes edge scores.
            train_pos_u (Tensor): Source nodes of positive training edges.
            train_pos_v (Tensor): Destination nodes of positive training edges.
            train_neg_u (Tensor): Source nodes of negative training edges.
            train_neg_v (Tensor): Destination nodes of negative training edges.
            g (DGLGraph): The homogeneous protein-protein interaction graph.
            features (Tensor): Node feature matrix.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            loss_fn (torch.nn.Module): Loss function (e.g., BCEWithLogitsLoss).

        Returns:
            None: Trains the model in-place.
        """
        model.train()

        g = g.to(self.device)
        features = g.ndata['feat'].to(self.device)
        labels = labels.to(self.device)
        train_idx = train_idx.to(self.device)
        val_idx = val_idx.to(self.device)

        losses = []
        val_accuracy = []
        val_f1s = []
        val_precisions = []
        val_recalls = []

        best_f1 = -1
        best_model_state = None
        best_threshold = 0.5  # Default fallback

        for epoch in range(self.epochs):
            logits = model(g, features)
            loss = loss_fn(logits[train_idx], labels[train_idx])
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc, f1, prec, rec, _, current_thresh = self.evaluating_model(
                model, g, labels, val_idx
            )
            if f1 > best_f1:
                best_f1 = f1
                best_model_state = model.state_dict()
                best_threshold = current_thresh
            val_accuracy.append(acc)
            val_f1s.append(f1)
            val_precisions.append(prec)
            val_recalls.append(rec)
            if epoch % 5 == 0:
                print(
                    f"Epoch {epoch}, Loss {loss.item():.4f}"
                    f", Val Acc {acc:.4f}, Val F1 {f1:.4f}"
                    f", Val Prec {prec:.4f}, Val Rec {rec:.4f}"
                )
        print(f"Best Threshold: {best_threshold:.2f}")
        model.load_state_dict(best_model_state)
        return losses, val_accuracy, val_f1s, val_precisions, val_recalls, best_threshold

    def evaluating_model(self, model, g, labels, idx, threshold=None):
        """
        Evaluates the trained model on test edges and computes accuracy.

        Args:
            model (nn.Module): The GNN model.
            predictor (nn.Module): The predictor module for edge scores.
            test_pos_u (Tensor): Source nodes of positive test edges.
            test_pos_v (Tensor): Destination nodes of positive test edges.
            test_neg_u (Tensor): Source nodes of negative test edges.
            test_neg_v (Tensor): Destination nodes of negative test edges.
            g (DGLGraph): The homogeneous graph.
            features (Tensor): Node feature matrix.

        Returns:
            tuple: (predictions, u_test, v_test)
                - predictions (Tensor): Binary predictions for test edges.
                - u_test (Tensor): Source nodes for test edges.
                - v_test (Tensor): Destination nodes for test edges.
        """
        model.eval()
        with torch.no_grad():
            g = g.to(self.device)
            features = g.ndata['feat'].to(self.device)
            labels = labels.to(self.device)
            idx = idx.to(self.device)

            logits = model(g, features)
            probs = torch.softmax(logits[idx], dim=1)
            y_scores = probs[:, 1].to(self.device)
            y_true = labels[idx].to(self.device)

            if threshold is None:
                threshold = self.evaluate_threshold_sweep(y_true, y_scores)
            else:
                print(f"Using fixed threshold = {threshold:.2f}")

            preds = (y_scores > threshold).long()
            acc = (preds == y_true).float().mean().item()
            f1 = f1_score(y_true, preds, average='binary', zero_division=0)
            precision = precision_score(y_true, preds, zero_division=0)
            recall = recall_score(y_true, preds, zero_division=0)
            return acc, f1, precision, recall, preds, threshold

    def evaluate_threshold_sweep(self, y_true, y_scores):
        thresholds = [i / 100 for i in range(5, 96, 5)]  # from 0.05 to 0.95
        best_f1 = -1
        best_threshold = 0.05

        for t in thresholds:
            y_pred = (y_scores > t).long()
            f1 = f1_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)

            if rec >= 0.7 and f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        return best_threshold

    def obtaining_predicted_proteins(self, node_index, preds, val_idx, seed_nodes):
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

    def main(self):
        """
        Executes the full pipeline for link prediction:
            - Loads and processes data.
            - Builds homogeneous graph.
            - Splits edges into train/test sets.
            - Trains a GNN model.
            - Evaluates model performance.
            - Extracts and saves predicted and real PPIs for each disease.

        Returns:
            None
        """
        df_pro_pro, df_gen_pro, df_dis_gen, df_dis_pro, self.selected_diseases = self.DC.main()
        G_ppi = self.HomoGraph.create_graph(df_pro_pro)
        pr_scores = self.HomoGraph.get_protein_pagerank(G_ppi)
        disease_pro_mapping = mapping_diseases_to_proteins(df_dis_pro)
        node_index = {node: i for i, node in enumerate(list(G_ppi.nodes()))}
        all_proteins = set(G_ppi.nodes())
        for disease in self.selected_diseases:
            print('\nDisease of interest: ', disease)
            node_scoring = disease_pro_mapping[disease]
            seed_nodes = {key for key, _ in node_scoring.items()}
            seed_nodes = seed_nodes.intersection(G_ppi.nodes())

            known_proteins = seed_nodes
            candidates = [p for p in all_proteins if p not in known_proteins]
            candidates = sorted(candidates, key=lambda p: pr_scores.get(p, 0), reverse=True)
            sampled_negatives = candidates[:min(10 * len(known_proteins), len(candidates))]

            expanded_nodes = list(seed_nodes) + sampled_negatives
            labels = generate_expanded_labels(
                seed_nodes, {k: i for i, k in enumerate(expanded_nodes)}, expanded_nodes
            )

            g = self.HomoGraph.convert_networkx_to_dgl_graph(
                 G_ppi.subgraph(expanded_nodes).copy(),
                 {k: node_scoring.get(k, 0.0) for k in expanded_nodes},
                 {k: i for i, k in enumerate(expanded_nodes)},
                 expanded_nodes,
                 pr_scores
            )
            print(f"# total labels: {len(labels)}")
            print(f"Labels == 1: {(labels == 1).sum().item()}")
            print(f"Labels == 0: {(labels == 0).sum().item()}")
            train_idx, val_idx, test_idx = split_train_test_val_indices(
                labels
            )
            # labels = generate_labels(seed_nodes, node_index, g.num_nodes())
            # train_idx, val_idx, test_idx = split_train_test_val_indices(
            #     node_index, train_ratio=0.7, val_ratio=0.15
            # )

            # Prepare model and optimizer
            model = GNN(in_feats=g.ndata['feat'].shape[1], hidden_feats=64).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

            # Compute class counts
            num_0 = (labels[train_idx] == 0).sum().item()
            num_1 = (labels[train_idx] == 1).sum().item()
            # Create class weights: higher for underrepresented class 1
            weights = torch.tensor([1.0, (num_0 / num_1)*0.5], dtype=torch.float32)
            loss_fn = nn.CrossEntropyLoss(weight=weights.to(self.device))
            loss_fn = FocalLoss()

            # Training loop
            losses, _, val_f1s, val_precisions, val_recalls, best_threshold = self.training_loop(
                model, g, labels, train_idx, val_idx, optimizer, loss_fn
            )

            os.makedirs(f'{self.output_path}/{disease}', exist_ok=True)
            plot_loss_and_metrics(
                losses, val_f1s, val_recalls, val_precisions,
                save_path=f"{self.output_path}/{disease}"
                "/training_progress.png"
            )

            # Evaluation
            test_acc, test_f1, test_prec, test_rec, preds, _ = self.evaluating_model(
                model, g, labels, test_idx, threshold=best_threshold
            )
            print(f"\nTest Accuracy: {test_acc:.4f}")
            print(f"Test F1: {test_f1:.4f}")
            print(f"Test Precision: {test_prec:.4f}")
            print(f"Test Recall: {test_rec:.4f}")
            counts = torch.bincount(preds)
            if len(counts) == 1:
                counts = torch.cat((counts, torch.tensor([0])))
            print('\nTest set size: ', len(test_idx))
            print(f"Test predictions breakdown: Predicted {counts[0]} as 0 "
                  f"(no seed nodes) and {counts[1]} as 1 (seed nodes)")

            y_true = labels[test_idx].to(self.device)
            y_pred = preds.to(self.device)

            plot_confusion_matrix(
                y_true, y_pred, save_path=f"{self.output_path}/{disease}"
                "/confusion_matrix.png"
            )

            logits = model(g, g.ndata['feat']).detach().to(self.device)
            y_scores = logits[test_idx.to(self.device), 1]

            plot_precision_recall_curve(
                y_true, y_scores,
                save_path=f"{self.output_path}/{disease}/pr_curve.png"
            )

            predicted_proteins = self.obtaining_predicted_proteins(
                node_index, preds, test_idx, seed_nodes
            )

            # Save predicted PPIs to a .txt file
            predicted_proteins.to_csv(
                f"{self.output_path}/{disease}"
                f"/predicted_proteins.txt", sep="\t", index=False
            )

            # Save real PPIs to a .txt file
            with open(
                f"{self.output_path}/{disease}/real_seeds_proteins.txt", "w"
            ) as f:
                for seed in seed_nodes:
                    f.write(f"{seed}\n")


if __name__ == "__main__":
    Main().main()
