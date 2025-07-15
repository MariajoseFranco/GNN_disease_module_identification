import os
import warnings

import torch
from evaluator import evaluating_model
from hyperparameter_tunning import run_optuna_tuning
from trainer import training_loop

from data_compilation import DataCompilation
from LinkPrediction.heterogeneous_graph import HeterogeneousGraph
from LinkPrediction.utils.build_model import builder
from LinkPrediction.utils.formatting_results import (
    displaying_metrics, obtaining_dis_pro_predicted, saving_results)
from LinkPrediction.utils.helper_functions import getting_seed_nodes
from LinkPrediction.utils.mappings import (mapping_dis_pro_edges_to_scores,
                                           mapping_to_encoded_ids)
from LinkPrediction.utils.splits import (neg_train_test_split,
                                         pos_train_test_split, splitting_sizes)
from utils import load_config
from visualizations import (plot_confusion_matrix, plot_loss_and_metrics,
                            plot_precision_recall_curve, plot_roc_curve,
                            visualize_disease_protein_associations)

warnings.filterwarnings("ignore")


class LinkPredictionPipeline():
    def __init__(self):
        # Paths
        self.config = load_config()
        self.data_path = self.config['data_dir']
        self.disease_path = self.config['disease_dir']
        self.output_path = self.config['results_linkpred_dir']
        os.makedirs(f'{self.output_path}', exist_ok=True)

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
        df_pro_pro, _, _, df_dis_pro, self.selected_diseases = self.DC.main()
        seed_edge_scores = mapping_dis_pro_edges_to_scores(df_dis_pro)
        G_dispro = self.HeteroGraph.create_graph(df_dis_pro, df_pro_pro)
        diseases_to_encoded_ids, proteins_to_encoded_ids = mapping_to_encoded_ids(
            df_dis_pro, df_pro_pro
        )

        # Get all edges of the forward edge type
        edge_type = ('disease', 'associates', 'protein')
        u, v, fwd_seed_score_tensor = self.HeteroGraph.obtaining_edges_and_score_edges(
            G_dispro, seed_edge_scores, edge_type
        )
        G_dispro = self.HeteroGraph.assigining_seed_score_to_edges(
            G_dispro, fwd_seed_score_tensor, edge_type, self.device
        )

        # Get all edges of the reverse edge type
        rev_edge_type = ('protein', 'rev_associates', 'disease')
        rev_u, rev_v, rev_seed_score_tensor = self.HeteroGraph.obtaining_edges_and_score_edges(
            G_dispro, seed_edge_scores, rev_edge_type
        )
        G_dispro = self.HeteroGraph.assigining_seed_score_to_edges(
            G_dispro, rev_seed_score_tensor, rev_edge_type, self.device
        )

        # Get and assign node features
        G_dispro, features = self.HeteroGraph.assigining_nodes_features(
            G_dispro, self.feat_dim, self.device
        )

        # Plot the heterogeneous graph
        visualize_disease_protein_associations(
            G_dispro,
            diseases=list(diseases_to_encoded_ids.keys()),
            max_edges=500,
            output_path=f"{self.output_path}/association_graph.png"
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

        best_trial = run_optuna_tuning(
            train_pos_g,
            train_neg_g,
            train_g,
            val_pos_g,
            val_neg_g,
            edge_type,
            features,
            self.output_path,
            all_etypes=etypes
        )
        best_params = best_trial.params

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

        seed_nodes = getting_seed_nodes(df_dis_pro, self.selected_diseases)

        predicted_dis_pro = obtaining_dis_pro_predicted(
            test_preds, test_pos_u, test_pos_v, test_neg_u, test_neg_v,
            diseases_to_encoded_ids, proteins_to_encoded_ids, seed_nodes
        )

        saving_results(
            predicted_dis_pro, best_params, best_threshold, test_probs, test_preds,
            labels, model, test_pos_u, test_pos_v, test_neg_u, test_neg_v, test_prec,
            test_rec, test_f1, test_auc, test_acc, diseases_to_encoded_ids,
            proteins_to_encoded_ids, self.output_path
        )
