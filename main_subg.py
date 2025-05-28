import itertools
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn

from data_compilation import DataCompilation
from dot_predictor_subg import DotPredictor
from GNN_sage import GNN
from graph_creation import GraphPPI
from utils import load_config, mapping_diseases_to_proteins
from utils import neg_train_test_split_subg as neg_train_test_split
from utils import pos_train_test_split

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
        self.predictor = DotPredictor()
        self.epochs = 100

    def training_loop(
            self,
            model,
            predictor,
            train_pos_u,
            train_pos_v,
            train_neg_u,
            train_neg_v,
            g,
            features,
            optimizer,
            loss_fn
    ):
        for epoch in range(self.epochs):
            model.train()

            h = model(g, features)

            u_train = torch.cat([train_pos_u, train_neg_u])
            v_train = torch.cat([train_pos_v, train_neg_v])
            labels = torch.cat([torch.ones(len(train_pos_u)), torch.zeros(len(train_neg_u))])

            logits = predictor(g, h, u_train, v_train)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0:
                print("In epoch {}, loss: {}".format(epoch, loss))

    def evaluating_model(
            self, model, predictor, test_pos_u, test_pos_v, test_neg_u, test_neg_v, g, features
    ):
        model.eval()
        with torch.no_grad():
            h = model(g, features)
            u_test = torch.cat([test_pos_u, test_neg_u])
            v_test = torch.cat([test_pos_v, test_neg_v])
            test_labels = torch.cat([torch.ones(len(test_pos_u)), torch.zeros(len(test_neg_u))])
            test_logits = predictor(g, h, u_test, v_test)

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
        diseases = self.DC.get_diseases()
        df_pro_pro, df_gen_pro, df_dis_gen, df_dis_pro, self.selected_diseases = self.DC.main(
            diseases
        )
        G_ppi = self.GPPI.create_homogeneous_graph(df_pro_pro)
        disease_pro_mapping = mapping_diseases_to_proteins(df_dis_pro)
        node_list = list(G_ppi.nodes())
        node_index = {node: i for i, node in enumerate(node_list)}
        for disease in self.selected_diseases:
            print('\nDisease of interest: ', disease)
            seed_nodes = disease_pro_mapping[disease]
            real_ppi_u = df_pro_pro[df_pro_pro['prA'].isin(seed_nodes)]
            real_ppi_v = df_pro_pro[df_pro_pro['prB'].isin(seed_nodes)]
            real_ppi = pd.concat([real_ppi_u, real_ppi_v])
            real_ppi.drop_duplicates()
            g = self.GPPI.convert_networkx_to_dgl_graph(G_ppi, seed_nodes)
            features = g.ndata['feat']
            u, v = g.edges()

            # Define test - train size sets
            eids = np.arange(g.num_edges())
            eids = np.random.permutation(eids)
            test_size = int(len(eids) * 0.2)

            # Positive edges (real)
            train_pos_u, train_pos_v, test_pos_u, test_pos_v = pos_train_test_split(
                u, v, eids, test_size
            )

            # Negative edges
            train_neg_u, train_neg_v, test_neg_u, test_neg_v = neg_train_test_split(
                u, v, g, test_size
            )

            # Prepare model and optimizer
            model = GNN(in_feats=features.shape[1], hidden_feats=64)
            optimizer = torch.optim.Adam(
                itertools.chain(model.parameters(), self.predictor.parameters()), lr=0.01
            )
            loss_fn = nn.BCEWithLogitsLoss()

            # Convert edge sets to tensors
            train_pos_u, train_pos_v, test_pos_u, test_pos_v = self.GPPI.convert_to_tensors(
                train_pos_u, train_pos_v, test_pos_u, test_pos_v
            )
            train_neg_u, train_neg_v, test_neg_u, test_neg_v = self.GPPI.convert_to_tensors(
                train_neg_u, train_neg_v, test_neg_u, test_neg_v
            )

            # Training loop
            self.training_loop(
                model,
                self.predictor,
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
                model, self.predictor, test_pos_u, test_pos_v, test_neg_u, test_neg_v, g, features
            )
            predicted_ppis = self.obtaining_ppi_predicted(
                node_index, preds, u_test, v_test
            )

            # Save predicted PPIs to a .txt file
            with open(f"{self.output_path}/predicted_ppis_{disease}.txt", "w") as f:
                for u, v in predicted_ppis:
                    f.write(f"{u}\t{v}\n")

            # Save real PPIs to a .txt file
            real_ppi.to_csv(f"{self.output_path}/real_ppis_{disease}.txt", sep="\t", index=False)


if __name__ == "__main__":
    Main().main()
