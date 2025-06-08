import itertools
import warnings

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from data_compilation import DataCompilation
from dot_predictor import DotPredictor
from graph_creation import GraphPPI
from heteroGNN import HeteroGNN as GNN
from utils import (load_config, mapping_index_to_node, neg_train_test_split,
                   pos_train_test_split,
                   visualize_disease_protein_associations)

warnings.filterwarnings("ignore")


class Main():
    def __init__(self):
        # Paths
        self.config = load_config()
        self.data_path = self.config['data_dir']
        self.disease_path = self.config['disease_dir']
        self.output_path = self.config['results_dir']

        self.DC = DataCompilation(self.data_path, self.disease_path)
        self.GPPI = GraphPPI()
        self.epochs = 100

    def training_loop(
            self,
            model,
            train_pos_g,
            train_neg_g,
            train_g,
            val_pos_g,
            val_neg_g,
            features,
            optimizer,
            pred,
            edge_type
    ):
        """
        Trains the GNN model on a heterogeneous graph for link prediction.

        Args:
            model (nn.Module): The GNN model.
            train_pos_g (DGLGraph): Graph containing positive training edges.
            train_neg_g (DGLGraph): Graph containing negative training edges.
            train_g (DGLGraph): The full graph (minus test edges) for message passing.
            features (dict): Node feature dictionary by node type.
            optimizer (torch.optim.Optimizer): Optimizer for the model.
            pred (DotPredictor): Predictor module to compute edge scores.
            edge_type (tuple): The edge type to predict (source, relation, target).

        Returns:
            dict: Node embeddings after training.
        """
        best_auc = 0
        best_state = None
        for epoch in range(self.epochs):
            # forward
            h = model(train_g, features)
            pos_score = pred(train_pos_g, h, etype=edge_type, use_seed_score=True)
            neg_score = pred(train_neg_g, h, etype=edge_type, use_seed_score=True)
            loss = self.compute_loss(pos_score, neg_score)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0:
                val_auc, _ = self.compute_auc(
                    pred(val_pos_g, h, etype=edge_type),
                    pred(val_neg_g, h, etype=edge_type)
                )
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}")

                if val_auc > best_auc:
                    best_auc = val_auc
                    best_state = model.state_dict()
        model.load_state_dict(best_state)  # Restore best model
        return h

    def evaluating_model(self, test_pos_g, test_neg_g, pred, h, edge_type):
        """
        Evaluates the trained model using AUC on the test set.

        Args:
            test_pos_g (DGLGraph): Graph with positive test edges.
            test_neg_g (DGLGraph): Graph with negative test edges.
            pred (DotPredictor): Predictor module to compute edge scores.
            h (dict): Node embeddings from the trained model.
            edge_type (tuple): The edge type for prediction.

        Returns:
            tuple: (pos_score, neg_score, labels)
                - pos_score (Tensor): Scores for positive test edges.
                - neg_score (Tensor): Scores for negative test edges.
                - labels (ndarray): Ground truth labels (1 for positive, 0 for negative).
        """
        with torch.no_grad():
            pos_score = pred(test_pos_g, h, etype=edge_type, use_seed_score=False)
            neg_score = pred(test_neg_g, h, etype=edge_type, use_seed_score=False)
            auc, labels = self.compute_auc(pos_score, neg_score)
            print(f"Test AUC: {auc:.4f}")
            return pos_score, neg_score, labels

    def compute_loss(self, pos_score, neg_score):
        """
        Computes binary cross-entropy loss for link prediction.

        Args:
            pos_score (Tensor): Model scores for positive edges.
            neg_score (Tensor): Model scores for negative edges.

        Returns:
            Tensor: The binary cross-entropy loss.
        """
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
        )
        return F.binary_cross_entropy_with_logits(scores, labels)

    def compute_auc(self, pos_score, neg_score):
        """
        Computes the AUC metric for link prediction.

        Args:
            pos_score (Tensor): Model scores for positive edges.
            neg_score (Tensor): Model scores for negative edges.

        Returns:
            tuple: (auc, labels)
                - auc (float): The computed AUC score.
                - labels (ndarray): Ground truth labels for the test edges.
        """
        scores = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
        labels = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
        ).cpu().numpy()
        return roc_auc_score(labels, scores), labels

    def obtaining_dis_pro_predicted(
            self,
            pos_score,
            neg_score,
            test_pos_u,
            test_pos_v,
            test_neg_u,
            test_neg_v,
            disease_index_to_node,
            protein_index_to_node,
            seed_nodes
    ):
        """
        Maps predicted edges from indices to disease-protein pairs
        and annotates with seed information.

        Args:
            pos_score (Tensor): Scores for positive test edges.
            neg_score (Tensor): Scores for negative test edges.
            test_pos_u (Tensor): Source nodes of positive test edges.
            test_pos_v (Tensor): Target nodes of positive test edges.
            test_neg_u (Tensor): Source nodes of negative test edges.
            test_neg_v (Tensor): Target nodes of negative test edges.
            disease_index_to_node (dict): Mapping from node indices to disease names.
            protein_index_to_node (dict): Mapping from node indices to protein IDs.
            seed_nodes (dict): Mapping from disease names to their known seed proteins.

        Returns:
            pd.DataFrame: DataFrame of predicted disease-protein
            associations with seed node annotation.
        """
        # Convert logits to probabilities
        all_scores = torch.cat([pos_score, neg_score])
        probs = torch.sigmoid(all_scores)

        # Binary prediction with threshold 0.5
        preds = (probs >= 0.5).int()

        # Concatenate test edges
        all_u = torch.cat([test_pos_u, test_neg_u])
        all_v = torch.cat([test_pos_v, test_neg_v])

        # Select predicted positive edges
        predicted_indices = (preds == 1).nonzero(as_tuple=True)[0]
        predicted_edges = [(all_u[i].item(), all_v[i].item()) for i in predicted_indices]
        predicted_named_edges = [
            (disease_index_to_node[u], protein_index_to_node[v])
            for u, v in predicted_edges
        ]
        predicted_df = pd.DataFrame(predicted_named_edges, columns=['disease', 'protein'])
        predicted_df['is_seed_node'] = False  # Initialize the column
        # Sort by disease
        predicted_df = predicted_df.sort_values(by='disease').reset_index(drop=True)
        for disease, seeds in seed_nodes.items():
            mask = predicted_df['disease'] == disease
            predicted_df.loc[mask, 'is_seed_node'] = predicted_df.loc[mask, 'protein'].isin(seeds)
        return predicted_df

    def main(self):
        """
        Executes the full pipeline for link prediction using a heterogeneous graph:
            - Loads and processes data.
            - Builds a heterogeneous disease-protein interaction graph.
            - Visualizes disease-protein associations.
            - Splits edges into train/test sets (positive and negative samples).
            - Trains a GNN model for link prediction.
            - Evaluates model performance using AUC.
            - Maps predicted disease-protein associations to node names.
            - Marks if predicted proteins are seed nodes.
            - Saves predicted associations as a .txt file.

        Returns:
            None
        """
        df_pro_pro, df_gen_pro, df_dis_gen, df_dis_pro = self.DC.main()
        df_dis_pro, self.selected_diseases = self.DC.get_matched_diseases(
            df_dis_pro, self.output_path
        )
        seed_edge_scores = {(row['disease_id'], row['protein_id_enc']): row['score']
                            for idx, row in df_dis_pro.iterrows()}
        G_dispro = self.GPPI.create_heterogeneous_graph(df_dis_pro, df_pro_pro)

        disease_index_to_node, protein_index_to_node = mapping_index_to_node(
            df_dis_pro, df_pro_pro
        )
        edge_type = ('disease', 'associates', 'protein')
        u, v = G_dispro.edges(etype=edge_type)
        # Get all edges of the reverse type
        rev_edge_type = ('protein', 'rev_associates', 'disease')
        rev_u, rev_v = G_dispro.edges(etype=rev_edge_type)

        num_edges = G_dispro.num_edges(etype=edge_type)
        seed_score_tensor = torch.zeros(num_edges, 1)

        for i, (src, dst) in enumerate(zip(u.tolist(), v.tolist())):
            if (src, dst) in seed_edge_scores:
                seed_score_tensor[i] = seed_edge_scores[(src, dst)]

        G_dispro.edges[edge_type].data['seed_score'] = seed_score_tensor
        visualize_disease_protein_associations(
            G_dispro, diseases=list(disease_index_to_node.keys()), max_edges=500
        )

        # Define test - train size sets
        eids = np.arange(G_dispro.num_edges(etype=edge_type))
        eids = np.random.permutation(eids)

        train_size = int(0.7 * len(eids))
        test_size = int(0.15 * len(eids))
        val_size = int(0.15 * len(eids))

        # Positive edges (real)
        train_pos_u, train_pos_v, val_pos_u, val_pos_v, test_pos_u, test_pos_v, val_eids, test_eids = (
            pos_train_test_split(u, v, eids, train_size, val_size, test_size)
        )

        # Negative edges
        train_neg_u, train_neg_v, val_neg_u, val_neg_v, test_neg_u, test_neg_v = (
            neg_train_test_split(G_dispro, edge_type, train_size, val_size, test_size)
        )

        # Identify indices of reverse edges that match
        val_pairs_set = set(zip(val_pos_v.tolist(), val_pos_u.tolist()))
        test_pairs_set = set(zip(test_pos_v.tolist(), test_pos_u.tolist()))

        # Build mask for reverse edge indices to remove
        rev_val_eids_to_remove = [
            i for i, (src, dst) in enumerate(zip(rev_u.tolist(), rev_v.tolist()))
            if (src, dst) in val_pairs_set
        ]
        rev_test_eids_to_remove = [
            i for i, (src, dst) in enumerate(zip(rev_u.tolist(), rev_v.tolist()))
            if (src, dst) in test_pairs_set
        ]
        rev_eids_to_remove = np.concatenate(
            [rev_val_eids_to_remove, rev_test_eids_to_remove]
        )

        # Remove test edges from both directions
        G_tmp = dgl.remove_edges(G_dispro, np.concatenate([val_eids, test_eids]), etype=edge_type)
        if len(rev_eids_to_remove) > 0:
            train_g = dgl.remove_edges(G_tmp, rev_eids_to_remove, etype=rev_edge_type)
        else:
            train_g = G_tmp
        feat_dim = 64
        G_dispro.nodes['disease'].data['feat'] = torch.randn(
            G_dispro.num_nodes('disease'), feat_dim
        )
        G_dispro.nodes['protein'].data['feat'] = torch.randn(
            G_dispro.num_nodes('protein'), feat_dim
        )
        features = {
            'disease': G_dispro.nodes['disease'].data['feat'],
            'protein': G_dispro.nodes['protein'].data['feat']
        }

        train_pos_g = self.GPPI.convert_to_heterogeneous_graph(
            G_dispro, edge_type, train_pos_u, train_pos_v
        )
        train_neg_g = self.GPPI.convert_to_heterogeneous_graph(
            G_dispro, edge_type, train_neg_u, train_neg_v
        )
        val_pos_g = self.GPPI.convert_to_heterogeneous_graph(
            G_dispro, edge_type, val_pos_u, val_pos_v
        )
        val_neg_g = self.GPPI.convert_to_heterogeneous_graph(
            G_dispro, edge_type, val_neg_u, val_neg_v
        )
        test_pos_g = self.GPPI.convert_to_heterogeneous_graph(
            G_dispro, edge_type, test_pos_u, test_pos_v
        )
        test_neg_g = self.GPPI.convert_to_heterogeneous_graph(
            G_dispro, edge_type, test_neg_u, test_neg_v
        )

        # Prepare model and optimizer
        in_feats = features['disease'].shape[1]
        model = GNN(in_feats=in_feats, hidden_feats=feat_dim)
        pred = DotPredictor()
        optimizer = torch.optim.Adam(
            itertools.chain(model.parameters(), pred.parameters()), lr=0.01
        )

        # Training loop
        h = self.training_loop(
            model,
            train_pos_g,
            train_neg_g,
            train_g,
            val_pos_g,
            val_neg_g,
            features,
            optimizer,
            pred,
            edge_type
        )

        # Evaluation
        pos_score, neg_score, labels = self.evaluating_model(
            test_pos_g, test_neg_g, pred, h, edge_type
        )

        seed_nodes = {}
        for disease in self.selected_diseases:
            df = df_dis_pro[df_dis_pro['disease_name'] == disease]
            tuple_seed_nodes = tuple(df['protein_id'])
            seed_nodes[disease] = tuple_seed_nodes

        predicted_dis_pro = self.obtaining_dis_pro_predicted(
            pos_score, neg_score, test_pos_u, test_pos_v, test_neg_u, test_neg_v,
            disease_index_to_node, protein_index_to_node, seed_nodes
        )
        # Save predicted DISEASE-PROTEIN associations to a .txt file
        predicted_dis_pro.to_csv(f"{self.output_path}/predicted_dis_pro.txt", sep="\t", index=False)


if __name__ == "__main__":
    Main().main()
