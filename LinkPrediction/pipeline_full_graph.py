import os
import warnings

import torch
from evaluator import evaluating_model
from trainer import training_loop

from data_compilation import DataCompilation
from LinkPrediction.heterogeneous_graph import HeterogeneousGraph
from LinkPrediction.utils.build_model import builder
from LinkPrediction.utils.formatting_results import (
    displaying_metrics, obtaining_dis_dru_predicted, saving_results_full_graph)
from LinkPrediction.utils.splits import (neg_train_test_split,
                                         pos_train_test_split, splitting_sizes)
from utils import load_config
from visualizations import (plot_confusion_matrix, plot_loss_and_metrics,
                            plot_precision_recall_curve, plot_roc_curve)

warnings.filterwarnings("ignore")


class LinkPredictionFullGraphPipeline():
    def __init__(self):
        # Paths
        self.config = load_config()
        self.data_path = self.config['data_dir']
        self.disease_path = self.config['disease_dir']
        self.output_path = self.config['results_linkpred_dir']
        os.makedirs(f'{self.output_path}', exist_ok=True)
        os.makedirs(f"{self.output_path}/drugs", exist_ok=True)
        self.output_path = f"{self.output_path}/drugs"

        self.DC = DataCompilation(self.data_path, self.disease_path, self.output_path)
        self.HeteroGraph = HeterogeneousGraph()
        self.epochs = 200
        self.feat_dim = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        """
        Executes the full pipeline for link prediction over a heterogeneous disease-protein graph.

        Pipeline steps:
            - Loads and processes input data.
            - Builds a heterogeneous DGL graph combining disease-protein and protein-protein
            interactions.
            - Assigns seed scores and random features to nodes.
            - Visualizes disease-protein associations in the graph.
            - Splits the graph edges into train, validation, and test sets (positive and negative).
            - Performs hyperparameter tuning using Optuna.
            - Builds and trains a Heterogeneous GNN model for link prediction.
            - Evaluates model performance using standard metrics
            (AUC, F1, Precision, Recall, Accuracy).
            - Maps predicted disease-protein edges back to their original names.
            - Annotates predictions as seed or non-seed associations.
            - Saves predictions, metrics, and trained model.

        Returns:
            None
        """
        (
            df_ddi_dru, df_ddi_phe, df_dis_dru_the,
            df_dis_pat, df_dis_pro, df_dis_sym,
            df_dru_dru, df_dru_pro, df_dru_sym_ind,
            df_dru_sym_sef, df_pro_pat, df_pro_pro
        ) = self.DC.get_full_graph_data()

        edge_specs = [
            ('drug', 'ddi_dru', 'drug', df_ddi_dru['ddi'], df_ddi_dru['dru']),
            ('drug', 'ddi_phe', 'phenotype', df_ddi_phe['ddi'], df_ddi_phe['phe']),
            ('disease', 'dis_dru_the', 'drug', df_dis_dru_the['dis'], df_dis_dru_the['dru']),
            ('disease', 'dis_pat', 'pathway', df_dis_pat['dis'], df_dis_pat['pat']),
            ('disease', 'dis_pro', 'protein', df_dis_pro['dis'], df_dis_pro['pro']),
            ('disease', 'dse_sym', 'phenotype', df_dis_sym['dis'], df_dis_sym['sym']),
            ('drug', 'druA_druB', 'drug', df_dru_dru['drA'], df_dru_dru['drB']),
            ('drug', 'dru_pro', 'protein', df_dru_pro['dru'], df_dru_pro['pro']),
            ('drug', 'dru_sym_ind', 'phenotype', df_dru_sym_ind['dru'], df_dru_sym_ind['sym']),
            ('drug', 'dru_sym_sef', 'phenotype', df_dru_sym_sef['dru'], df_dru_sym_sef['sym']),
            ('protein', 'pro_pat', 'pathway', df_pro_pat['pro'], df_pro_pat['pat']),
            ('protein', 'proA_proB', 'protein', df_pro_pro['prA'], df_pro_pro['prB']),
        ]
        G_dispro, node_maps, edges = self.HeteroGraph.create_heterograph_with_mapped_ids(edge_specs)

        # Get all forward edges
        edge_type = ('disease', 'dis_dru_the', 'drug')
        u, v = G_dispro.edges(etype=edge_type)

        # Get all edges of the reverse type
        rev_edge_type = ('drug', 'rev_dis_dru_the', 'disease')
        rev_u, rev_v = G_dispro.edges(etype=rev_edge_type)

        # Get and assign node features
        G_dispro, features = self.HeteroGraph.assigining_nodes_features_full_graph(
            G_dispro, self.feat_dim, self.device
        )

        # Splitting sets
        eids, train_size, val_size, test_size = splitting_sizes(G_dispro, edge_type)

        # Positive edges (real)
        (
            train_pos_u, train_pos_v,
            val_pos_u, val_pos_v,
            test_pos_u, test_pos_v,
            val_eids, test_eids
        ) = pos_train_test_split(u, v, eids, train_size, val_size, test_size)

        # Negative edges
        train_neg_u, train_neg_v, val_neg_u, val_neg_v, test_neg_u, test_neg_v = (
            neg_train_test_split(G_dispro, edge_type, train_size, val_size, test_size)
        )

        # Remove test edges from both directions
        train_g = self.HeteroGraph.removing_edges_from_graph(
            G_dispro, val_pos_u, val_pos_v, test_pos_u, test_pos_v,
            rev_u, rev_v, val_eids, test_eids, edge_type, rev_edge_type
        )

        # Obtaining train-test-val pos and neg graphs
        train_pos_g = self.HeteroGraph.convert_to_heterogeneous_graph(
            G_dispro, edge_type, train_pos_u, train_pos_v
        )
        train_neg_g = self.HeteroGraph.convert_to_heterogeneous_graph(
            G_dispro, edge_type, train_neg_u, train_neg_v
        )
        val_pos_g = self.HeteroGraph.convert_to_heterogeneous_graph(
            G_dispro, edge_type, val_pos_u, val_pos_v
        )
        val_neg_g = self.HeteroGraph.convert_to_heterogeneous_graph(
            G_dispro, edge_type, val_neg_u, val_neg_v
        )
        test_pos_g = self.HeteroGraph.convert_to_heterogeneous_graph(
            G_dispro, edge_type, test_pos_u, test_pos_v
        )
        test_neg_g = self.HeteroGraph.convert_to_heterogeneous_graph(
            G_dispro, edge_type, test_neg_u, test_neg_v
        )
        etypes = list(G_dispro.etypes)

        best_params = {
            "hidden_feats": 128,
            "num_layers": 2,
            "layer_type": "GraphConv",
            "aggregator_type": "mean",
            "dropout": 0.106,
            "predictor_type": "mlp",
            "lr": 0.00041,
            "weight_decay": 0.00081,
            "use_focal": False
        }

        # Prepare model and optimizer
        model, optimizer, loss_fn, pred = builder(
            best_params, features, edge_type, etypes, train_pos_g, train_neg_g, self.device
        )

        # Training loop
        h, loss, val_acc, val_f1, val_prec, val_rec, best_threshold = training_loop(
            self.epochs,
            model,
            train_pos_g,
            train_neg_g,
            train_g,
            val_pos_g,
            val_neg_g,
            features,
            optimizer,
            pred,
            edge_type,
            loss_fn,
            self.device
        )

        plot_loss_and_metrics(
            loss, val_f1, val_rec, val_prec, save_path=f"{self.output_path}/training_progress.png"
        )

        # Evaluation
        (
            pos_score,
            neg_score,
            labels,
            test_prec,
            test_rec,
            test_f1,
            test_auc,
            test_acc,
            _
        ) = evaluating_model(
            test_pos_g, test_neg_g, pred, h, edge_type, self.device, threshold=best_threshold
        )

        test_scores = torch.cat([pos_score, neg_score])
        test_probs = torch.sigmoid(test_scores)
        test_preds = (test_probs > best_threshold).long()

        displaying_metrics(test_acc, test_f1, test_rec, test_prec, test_auc, test_preds)

        plot_confusion_matrix(
            labels, test_preds, save_path=f"{self.output_path}/confusion_matrix.png"
        )

        plot_precision_recall_curve(
            labels, test_probs, save_path=f"{self.output_path}/precision_recall_curve.png"
        )

        plot_roc_curve(labels, test_probs, save_path=f"{self.output_path}/roc_curve.png")

        diseases_to_encoded_ids = {v: k for k, v in node_maps['disease'].items()}
        drugs_to_encoded_ids = {v: k for k, v in node_maps['drug'].items()}

        predicted_dis_dru = obtaining_dis_dru_predicted(
            pos_score, neg_score, test_pos_u, test_pos_v, test_neg_u, test_neg_v,
            diseases_to_encoded_ids, drugs_to_encoded_ids
        )

        saving_results_full_graph(
            predicted_dis_dru, best_params, best_threshold, test_probs, test_preds,
            labels, model, test_pos_u, test_pos_v, test_neg_u, test_neg_v, test_prec,
            test_rec, test_f1, test_auc, test_acc, diseases_to_encoded_ids,
            drugs_to_encoded_ids, self.output_path
        )
