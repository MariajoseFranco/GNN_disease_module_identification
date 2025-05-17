import itertools
import warnings

import dgl
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

    def obtaining_ppi_predicted(self, node_index, preds, u_test, v_test):
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

    def main(self):
        diseases = self.DC.get_diseases()
        df_pro_pro, df_gen_pro, df_dis_gen, df_dis_pro, self.selected_diseases = self.DC.main(
            diseases
        )
        G_dispro, G_ppi, disease_pro_mapping = self.GPPI.main(df_pro_pro, df_dis_pro)
        # node_list = list(G_ppi.nodes())
        # node_index = {node: i for i, node in enumerate(node_list)}

        # the selected diseases in this moment are 28,
        # we need to match the 70 diseases and use them all
        for disease in self.selected_diseases:
            print('\nDisease of interest: ', disease)
            seed_nodes = disease_pro_mapping[disease]
            real_ppi_u = df_pro_pro[df_pro_pro['prA'].isin(seed_nodes)]
            real_ppi_v = df_pro_pro[df_pro_pro['prB'].isin(seed_nodes)]
            real_ppi = pd.concat([real_ppi_u, real_ppi_v])
            real_ppi.drop_duplicates()
            # g_homogeneous = convert_to_dgl_graph(G_ppi, seed_nodes)
            g = G_dispro
            edge_type = ('disease', 'associates', 'protein')
            u, v = g.edges(etype=edge_type)

            # Define test - train size sets
            eids = np.arange(g.num_edges(etype=edge_type))
            eids = np.random.permutation(eids)
            test_size = int(len(eids) * 0.2)

            # Positive edges (real)
            train_pos_u, train_pos_v, test_pos_u, test_pos_v = self.pos_train_test_split(
                u, v, eids, test_size
            )

            # Negative edges
            train_neg_u, train_neg_v, test_neg_u, test_neg_v = self.sample_negative_edges(
                g, edge_type, num_samples=len(train_pos_u), test_size=test_size
            )
            # train_neg_u, train_neg_v, test_neg_u, test_neg_v = self.neg_train_test_split(
            #     u, v, g, test_size
            # )

            train_g = dgl.remove_edges(g, eids[:test_size], etype=edge_type)
            feat_dim = 64
            g.nodes['disease'].data['feat'] = torch.randn(g.num_nodes('disease'), feat_dim)
            g.nodes['protein'].data['feat'] = torch.randn(g.num_nodes('protein'), feat_dim)
            features = {
                'disease': g.nodes['disease'].data['feat'],
                'protein': g.nodes['protein'].data['feat']
            }

            train_pos_g = dgl.heterograph(
                {edge_type: (train_pos_u, train_pos_v)},
                num_nodes_dict={
                    'disease': g.num_nodes('disease'),
                    'protein': g.num_nodes('protein')
                }
            )
            train_neg_g = dgl.heterograph(
                {edge_type: (train_neg_u, train_neg_v)},
                num_nodes_dict={
                    'disease': g.num_nodes('disease'),
                    'protein': g.num_nodes('protein')
                }
            )

            test_pos_g = dgl.heterograph(
                {edge_type: (test_pos_u, test_pos_v)},
                num_nodes_dict={
                    'disease': g.num_nodes('disease'),
                    'protein': g.num_nodes('protein')
                }
            )
            test_neg_g = dgl.heterograph(
                {edge_type: (test_neg_u, test_neg_v)},
                num_nodes_dict={
                    'disease': g.num_nodes('disease'),
                    'protein': g.num_nodes('protein')
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
            # predicted_ppis = self.obtaining_ppi_predicted(
            #     node_index, preds, u_test, v_test
            # )

            # Save predicted PPIs to a .txt file
            # with open(f"outputs/predicted_ppis_{disease}.txt", "w") as f:
            #     for u, v in predicted_ppis:
            #         f.write(f"{u}\t{v}\n")

            # # Save real PPIs to a .txt file
            # real_ppi.to_csv(f"outputs/real_ppis_{disease}.txt", sep="\t", index=False)


if __name__ == "__main__":
    path = "./data/"
    Main(path).main()
