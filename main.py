import warnings

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from data_compilation import DataCompilation
from GNN import GNN
from graph_creation import GraphPPI
from utils import convert_to_dgl_graph

warnings.filterwarnings("ignore")


class Main():
    def __init__(self, path):
        # Select the diseases to work with
        self.selected_diseases = ["Albinism", "Alcohol Use Disorder"]
        self.DC = DataCompilation(path, self.selected_diseases)
        self.GPPI = GraphPPI()

    def pos_train_test_split(self, u, v, eids, test_size):
        test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
        train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]
        return train_pos_u, train_pos_v, test_pos_u, test_pos_v

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

    def convert_to_tensors(self, train_u, train_v, test_u, test_v):
        train_u = torch.tensor(train_u, dtype=torch.long)
        train_v = torch.tensor(train_v, dtype=torch.long)
        test_u = torch.tensor(test_u, dtype=torch.long)
        test_v = torch.tensor(test_v, dtype=torch.long)
        return train_u, train_v, test_u, test_v

    def training_loop(
            self,
            model,
            train_pos_u,
            train_pos_v,
            train_neg_u,
            train_neg_v,
            g,
            features,
            optimizer,
            loss_fn
    ):
        for epoch in range(100):
            model.train()

            # Combine pos/neg edges
            u_train = torch.cat([train_pos_u, train_neg_u])
            v_train = torch.cat([train_pos_v, train_neg_v])
            labels = torch.cat([torch.ones(len(train_pos_u)), torch.zeros(len(train_neg_u))])

            logits = model(g, features, u_train, v_train)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    def evaluating_model(self, model, test_pos_u, test_pos_v, test_neg_u, test_neg_v, g, features):
        model.eval()
        with torch.no_grad():
            # Combine pos/neg edges
            u_test = torch.cat([test_pos_u, test_neg_u])
            v_test = torch.cat([test_pos_v, test_neg_v])
            test_labels = torch.cat([torch.ones(len(test_pos_u)), torch.zeros(len(test_neg_u))])
            test_logits = model(g, features, u_test, v_test)

            preds = (torch.sigmoid(test_logits) > 0.5).float()
            acc = (preds == test_labels).float().mean().item()
            print(f"\nTest Accuracy: {acc:.4f}")
            return preds, u_test, v_test

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
        df_pro_pro, df_gen_pro, df_dis_gen, df_dis_pro = self.DC.main()
        G_ppi, disease_pro_mapping = self.GPPI.main(df_pro_pro, df_dis_pro)
        node_list = list(G_ppi.nodes())
        node_index = {node: i for i, node in enumerate(node_list)}
        for disease in self.selected_diseases:
            print('\nDisease of interest: ', disease)
            seed_nodes = disease_pro_mapping[disease]
            g = convert_to_dgl_graph(G_ppi, seed_nodes)
            features = g.ndata['feat']
            u, v = g.edges()

            # Define test - train size sets
            eids = np.arange(g.num_edges())
            eids = np.random.permutation(eids)
            test_size = int(len(eids) * 0.2)

            # Positive edges (real)
            train_pos_u, train_pos_v, test_pos_u, test_pos_v = self.pos_train_test_split(
                u, v, eids, test_size
            )

            # Negative edges
            train_neg_u, train_neg_v, test_neg_u, test_neg_v = self.neg_train_test_split(
                u, v, g, test_size
            )

            # Prepare model and optimizer
            model = GNN(in_feats=features.shape[1], hidden_feats=64)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            loss_fn = nn.BCEWithLogitsLoss()

            # Convert edge sets to tensors
            train_pos_u, train_pos_v, test_pos_u, test_pos_v = self.convert_to_tensors(
                train_pos_u, train_pos_v, test_pos_u, test_pos_v
            )
            train_neg_u, train_neg_v, test_neg_u, test_neg_v = self.convert_to_tensors(
                train_neg_u, train_neg_v, test_neg_u, test_neg_v
            )

            # Training loop
            self.training_loop(
                model,
                train_pos_u,
                train_pos_v,
                train_neg_u,
                train_neg_v,
                g,
                features,
                optimizer,
                loss_fn
            )

            # Evaluation
            preds, u_test, v_test = self.evaluating_model(
                model, test_pos_u, test_pos_v, test_neg_u, test_neg_v, g, features
            )
            predicted_ppis = self.obtaining_ppi_predicted(
                node_index, preds, u_test, v_test
            )

            # Save predicted PPIs to a .txt file
            with open(f"predicted_ppis_{disease}.txt", "w") as f:
                for u, v in predicted_ppis:
                    f.write(f"{u}\t{v}\n")


if __name__ == "__main__":
    path = "./data/"
    # path = "/app/data/"
    Main(path).main()
