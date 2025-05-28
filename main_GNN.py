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
from utils import load_config, mapping_index_to_node
from utils import neg_train_test_split_gnn as neg_train_test_split
from utils import pos_train_test_split, visualize_disease_protein_associations

warnings.filterwarnings("ignore")


class Main():
    def __init__(self):
        # Paths
        self.config = load_config()
        self.data_path = self.config['data_dir']
        self.disease_path = self.config['disease_txt']
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
            features,
            optimizer,
            pred,
            edge_type
    ):
        for epoch in range(self.epochs):
            # forward
            h = model(train_g, features)
            pos_score = pred(train_pos_g, h, edge_type)
            neg_score = pred(train_neg_g, h, edge_type)
            loss = self.compute_loss(pos_score, neg_score)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0:
                print("In epoch {}, loss: {}".format(epoch, loss))
        return h

    def evaluating_model(self, test_pos_g, test_neg_g, pred, h, edge_type):
        with torch.no_grad():
            pos_score = pred(test_pos_g, h, edge_type)
            neg_score = pred(test_neg_g, h, edge_type)
            auc, labels = self.compute_auc(pos_score, neg_score)
            print("AUC", auc)
            return pos_score, neg_score, labels

    def compute_loss(self, pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
        )
        return F.binary_cross_entropy_with_logits(scores, labels)

    def compute_auc(self, pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score]).numpy()
        labels = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
        ).numpy()
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
        diseases = self.DC.get_diseases()
        df_pro_pro, df_gen_pro, df_dis_gen, df_dis_pro, self.selected_diseases = self.DC.main(
            diseases
        )
        G_dispro = self.GPPI.create_heterogeneous_graph(df_dis_pro, df_pro_pro)
        visualize_disease_protein_associations(G_dispro, diseases=[0, 1, 2], max_edges=100)

        disease_index_to_node, protein_index_to_node = mapping_index_to_node(
            df_dis_pro, df_pro_pro
        )
        edge_type = ('disease', 'associates', 'protein')
        u, v = G_dispro.edges(etype=edge_type)

        # Define test - train size sets
        eids = np.arange(G_dispro.num_edges(etype=edge_type))
        eids = np.random.permutation(eids)
        test_size = int(len(eids) * 0.2)

        # Positive edges (real)
        train_pos_u, train_pos_v, test_pos_u, test_pos_v = pos_train_test_split(
            u, v, eids, test_size
        )

        # Negative edges
        train_neg_u, train_neg_v, test_neg_u, test_neg_v = neg_train_test_split(
            G_dispro, edge_type, num_samples=len(train_pos_u), test_size=test_size
        )

        train_g = dgl.remove_edges(G_dispro, eids[:test_size], etype=edge_type)
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
