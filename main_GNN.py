import itertools
import warnings

import dgl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from data_compilation import DataCompilation
from dot_predictor import DotPredictor
from graph_creation import GraphPPI
# from GNN_conv import GNN
# from GNN_sage import GNN
from heteroGNN import HeteroGNN as GNN

# from utils import convert_to_dgl_graph

warnings.filterwarnings("ignore")


class Main():
    def __init__(self, path):
        # Select the diseases to work with
        self.DC = DataCompilation(path)
        self.GPPI = GraphPPI()

    def visualize_disease_protein_associations(self, g, diseases, max_edges=200):
        # Only use 'associates' edge type
        etype = ('disease', 'associates', 'protein')
        src, dst = g.edges(etype=etype)

        # Filter edges for the selected disease nodes
        mask = torch.isin(src, torch.tensor(diseases))
        src = src[mask]
        dst = dst[mask]

        # Optionally limit number of edges
        if len(src) > max_edges:
            indices = torch.randperm(len(src))[:max_edges]
            src = src[indices]
            dst = dst[indices]

        # Build a NetworkX graph
        G_nx = nx.DiGraph()
        for s, d in zip(src.tolist(), dst.tolist()):
            disease_label = f"disease_{s}"
            protein_label = f"protein_{d}"
            G_nx.add_node(disease_label, bipartite=0)
            G_nx.add_node(protein_label, bipartite=1)
            G_nx.add_edge(disease_label, protein_label)

        # Layout and draw
        pos = nx.spring_layout(G_nx, k=0.5, seed=42)
        plt.figure(figsize=(16, 10))
        node_colors = ['lightcoral' if n.startswith('disease') else 'skyblue' for n in G_nx.nodes()]
        nx.draw(
            G_nx, pos,
            with_labels=True,
            node_color=node_colors,
            node_size=300,
            font_size=7,
            edge_color='gray',
            alpha=0.9
        )

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightcoral', label='Diseases'),
            Patch(facecolor='skyblue', label='Proteins')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        plt.title("Disease–Protein Associations")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def pos_train_test_split(self, u, v, eids, test_size):
        test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
        train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]
        return train_pos_u, train_pos_v, test_pos_u, test_pos_v

    def sample_negative_edges(self, g, etype, num_samples, test_size):
        src_type, _, dst_type = g.to_canonical_etype(etype)

        # Sample negative candidate edges
        src_ids = torch.randint(0, g.num_nodes(src_type), (num_samples,))
        dst_ids = torch.randint(0, g.num_nodes(dst_type), (num_samples,))

        # Ensure test_size < num_samples
        assert test_size < num_samples, "test_size must be smaller than num_samples"

        # Shuffle indices for splitting
        indices = np.random.permutation(num_samples)

        # Train/Test split
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]

        test_neg_u = src_ids[test_indices]
        test_neg_v = dst_ids[test_indices]
        train_neg_u = src_ids[train_indices]
        train_neg_v = dst_ids[train_indices]

        return train_neg_u, train_neg_v, test_neg_u, test_neg_v

    def neg_train_test_split(self, u, v, g, test_size):
        adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
        adj_neg = 1 - adj.todense() - np.eye(g.num_nodes())
        neg_u, neg_v = np.where(adj_neg != 0)

        neg_eids = np.random.choice(len(neg_u), g.num_edges())
        test_neg_u, test_neg_v = (
            neg_u[neg_eids[:test_size]],
            neg_v[neg_eids[:test_size]],
        )
        train_neg_u, train_neg_v = (
            neg_u[neg_eids[test_size:]],
            neg_v[neg_eids[test_size:]],
        )
        return train_neg_u, train_neg_v, test_neg_u, test_neg_v

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

    def convert_to_tensors(self, train_u, train_v, test_u, test_v):
        train_u = torch.tensor(train_u, dtype=torch.long)
        train_v = torch.tensor(train_v, dtype=torch.long)
        test_u = torch.tensor(test_u, dtype=torch.long)
        test_v = torch.tensor(test_v, dtype=torch.long)
        return train_u, train_v, test_u, test_v

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
        for epoch in range(100):
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

    def obtaining_ppi_predicted_old(self, node_index, preds, u_test, v_test):
        index_node = {i: n for n, i in node_index.items()}
        pred_positive_indices = (preds == 1).nonzero(as_tuple=True)[0]
        predicted_ppis = []
        for i in pred_positive_indices:
            u_idx = u_test[i].item()
            v_idx = v_test[i].item()
            protein_u = index_node.get(u_idx, u_idx)
            protein_v = index_node.get(v_idx, v_idx)
            predicted_ppis.append((protein_u, protein_v))
        return predicted_ppis

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

    def mapping_index_to_node(self, df_dis_pro, df_pro_pro):
        # Disease mapping
        disease_index_to_node = {
            idx: disease for idx, disease in df_dis_pro[['disease_id', 'disease_name']]
            .drop_duplicates()
            .values
        }

        # Protein mapping
        # Extract and unify mappings from all sources
        dis_pro_mapping = df_dis_pro[['protein_id_enc', 'protein_id']].drop_duplicates()
        src_mapping = df_pro_pro[['src_id', 'prA']].drop_duplicates()
        dst_mapping = df_pro_pro[['dst_id', 'prB']].drop_duplicates()

        # Rename columns for consistency
        src_mapping.columns = ['protein_id_enc', 'protein_id']
        dst_mapping.columns = ['protein_id_enc', 'protein_id']

        # Combine all mappings
        combined_mapping = pd.concat([dis_pro_mapping, src_mapping, dst_mapping]).drop_duplicates()

        # Create dictionary
        protein_index_to_node = {
            idx: protein for idx, protein in combined_mapping.values
        }
        return disease_index_to_node, protein_index_to_node

    def main(self):
        diseases = self.DC.get_diseases()
        df_pro_pro, df_gen_pro, df_dis_gen, df_dis_pro, self.selected_diseases = self.DC.main(
            diseases
        )
        G_dispro, G_ppi, disease_pro_mapping = self.GPPI.main(df_pro_pro, df_dis_pro)
        self.visualize_disease_protein_associations(G_dispro, diseases=[0, 1, 2], max_edges=100)

        disease_index_to_node, protein_index_to_node = self.mapping_index_to_node(
            df_dis_pro, df_pro_pro
        )
        # the selected diseases in this moment are 28,
        # we need to match the 70 diseases and use them all
        edge_type = ('disease', 'associates', 'protein')
        u, v = G_dispro.edges(etype=edge_type)

        # Define test - train size sets
        eids = np.arange(G_dispro.num_edges(etype=edge_type))
        eids = np.random.permutation(eids)
        test_size = int(len(eids) * 0.2)

        # Positive edges (real)
        train_pos_u, train_pos_v, test_pos_u, test_pos_v = self.pos_train_test_split(
            u, v, eids, test_size
        )

        # Negative edges
        train_neg_u, train_neg_v, test_neg_u, test_neg_v = self.sample_negative_edges(
            G_dispro, edge_type, num_samples=len(train_pos_u), test_size=test_size
        )
        # train_neg_u, train_neg_v, test_neg_u, test_neg_v = self.neg_train_test_split(
        #     u, v, g, test_size
        # )

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

        train_pos_g = dgl.heterograph(
            {edge_type: (train_pos_u, train_pos_v)},
            num_nodes_dict={
                'disease': G_dispro.num_nodes('disease'),
                'protein': G_dispro.num_nodes('protein')
            }
        )
        train_neg_g = dgl.heterograph(
            {edge_type: (train_neg_u, train_neg_v)},
            num_nodes_dict={
                'disease': G_dispro.num_nodes('disease'),
                'protein': G_dispro.num_nodes('protein')
            }
        )

        test_pos_g = dgl.heterograph(
            {edge_type: (test_pos_u, test_pos_v)},
            num_nodes_dict={
                'disease': G_dispro.num_nodes('disease'),
                'protein': G_dispro.num_nodes('protein')
            }
        )
        test_neg_g = dgl.heterograph(
            {edge_type: (test_neg_u, test_neg_v)},
            num_nodes_dict={
                'disease': G_dispro.num_nodes('disease'),
                'protein': G_dispro.num_nodes('protein')
            }
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
        predicted_dis_pro.to_csv("outputs/predicted_dis_pro.txt", sep="\t", index=False)


if __name__ == "__main__":
    path = "./data/"
    Main(path).main()
