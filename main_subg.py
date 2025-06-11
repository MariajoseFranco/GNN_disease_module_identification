import os
import warnings

import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn

from data_compilation import DataCompilation
from dot_predictor import DotPredictor
from GNN_sage import GNN
from graph_creation import GraphPPI
from utils import (generate_labels, load_config, mapping_diseases_to_proteins,
                   split_train_test_val_indices)

warnings.filterwarnings("ignore")


class Main():
    def __init__(self):
        # Paths
        self.config = load_config()
        self.data_path = self.config['data_dir']
        self.disease_path = self.config['disease_dir']
        self.output_path = self.config['results_dir']

        self.DC = DataCompilation(self.data_path, self.disease_path, self.output_path)
        self.GPPI = GraphPPI()
        self.predictor = DotPredictor()
        self.epochs = 100
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

        for epoch in range(self.epochs):
            logits = model(g, features)
            loss = loss_fn(logits[train_idx], labels[train_idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0:
                acc, f1, prec, rec, _ = self.evaluating_model(model, g, labels, val_idx)
                print(
                    f"Epoch {epoch}, Loss {loss.item():.4f}"
                    f", Val Acc {acc:.4f}, Val F1 {f1:.4f}"
                    f", Val Prec {prec:.4f}, Val Rec {rec:.4f}")

    def evaluating_model(self, model, g, labels, test_idx):
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
            test_idx = test_idx.to(self.device)

            logits = model(g, features)
            preds = logits[test_idx].argmax(dim=1).to(self.device)
            acc = (preds == labels[test_idx]).float().mean().item()
            f1 = f1_score(labels[test_idx], preds, average='binary', zero_division=0)
            precision = precision_score(labels[test_idx], preds, zero_division=0)
            recall = recall_score(labels[test_idx], preds, zero_division=0)
            return acc, f1, precision, recall, preds

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
        G_ppi = self.GPPI.create_homogeneous_graph(df_pro_pro)
        disease_pro_mapping = mapping_diseases_to_proteins(df_dis_pro)
        node_index = {node: i for i, node in enumerate(list(G_ppi.nodes()))}
        for disease in self.selected_diseases:
            print('\nDisease of interest: ', disease)
            node_scoring = disease_pro_mapping[disease]
            seed_nodes = {key for key, _ in node_scoring.items()}
            g = self.GPPI.convert_networkx_to_dgl_graph(
                G_ppi, node_scoring, node_index
            )
            labels = generate_labels(seed_nodes, node_index, g.num_nodes())
            train_idx, val_idx, test_idx = split_train_test_val_indices(
                node_index, train_ratio=0.7, val_ratio=0.15
            )

            # Prepare model and optimizer
            model = GNN(features=g.ndata['feat'].shape[1], hidden_feats=64).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            # Compute class counts
            num_0 = (labels[train_idx] == 0).sum().item()
            num_1 = (labels[train_idx] == 1).sum().item()
            # Create class weights: higher for underrepresented class 1
            weights = torch.tensor([1.0, num_0 / num_1], dtype=torch.float32)
            loss_fn = nn.CrossEntropyLoss(weight=weights.to(self.device))

            # Training loop
            self.training_loop(model, g, labels, train_idx, val_idx, optimizer, loss_fn)

            # Evaluation
            test_acc, test_f1, test_prec, test_rec, preds = self.evaluating_model(
                model, g, labels, test_idx
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
            predicted_proteins = self.obtaining_predicted_proteins(
                node_index, preds, test_idx, seed_nodes
            )

            # Save predicted PPIs to a .txt file
            os.makedirs(f'{self.output_path}/Subg Classifcation Task/{disease}', exist_ok=True)
            predicted_proteins.to_csv(
                f"{self.output_path}/Subg Classifcation Task/{disease}"
                f"/predicted_proteins.txt", sep="\t", index=False
            )

            # Save real PPIs to a .txt file
            with open(
                f"{self.output_path}/Subg Classifcation Task/{disease}/real_seeds_proteins.txt", "w"
            ) as f:
                for seed in seed_nodes:
                    f.write(f"{seed}\n")


if __name__ == "__main__":
    Main().main()
