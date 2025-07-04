import os
import warnings

import torch
from evaluator import evaluating_model
from hyperparameter_tunning import run_optuna_tuning
from trainer import training_loop

from data_compilation import DataCompilation
from SubgraphClassification.homogeneous_graph import HomogeneousGraph
from SubgraphClassification.utils.build_model import builder
from SubgraphClassification.utils.formatting_results import (
    displaying_metrics, obtaining_pred_scores, saving_results)
from SubgraphClassification.utils.labels import generate_expanded_labels
from SubgraphClassification.utils.mappings import (
    mapping_diseases_to_proteins, obtaining_predicted_proteins)
from SubgraphClassification.utils.splits import split_train_test_val_indices
from utils import load_config
from visualizations import (plot_all_curves, plot_confusion_matrix,
                            plot_loss_and_metrics, plot_precision_recall_curve,
                            plot_roc_curve)

warnings.filterwarnings("ignore")


class SubgPipeline():
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
        self.all_pr_curves = []
        self.all_roc_curves = []

    def run(self):
        """
        Executes the full Subgraph Node Classification pipeline for disease-protein
        module identification.

        This method iterates over a list of diseases and, for each one:
            - Loads the protein-protein interaction (PPI) graph and disease-protein associations.
            - Extracts seed and candidate nodes to form a subgraph.
            - Removes isolated nodes and generates node labels.
            - Converts the NetworkX subgraph into a DGL graph with engineered node features.
            - Splits nodes into train/validation/test subsets with class balancing.
            - Performs Optuna hyperparameter tuning on a GNN model.
            - Builds and trains the best GNN model configuration.
            - Evaluates the trained model using various classification metrics.
            - Saves predicted protein modules, evaluation plots, and serialized outputs.

        Additionally:
            - Plots and saves macro-averaged precision-recall and ROC curves across all diseases.

        Raises:
            FileNotFoundError: If input files specified in the configuration are missing.
            RuntimeError: If model training fails or GPU/CPU is unavailable.

        Returns:
            None
        """
        df_pro_pro, _, _, df_dis_pro, self.selected_diseases = self.DC.main()
        G_ppi = self.HomoGraph.create_graph(df_pro_pro)
        pr_scores = self.HomoGraph.get_protein_pagerank(G_ppi)
        disease_pro_mapping = mapping_diseases_to_proteins(df_dis_pro)
        all_proteins = set(G_ppi.nodes())
        for disease in self.selected_diseases:
            print('\nDisease of interest: ', disease)
            node_scoring = self.HomoGraph.get_node_scoring(
                disease_pro_mapping, disease
            )
            seed_nodes = self.HomoGraph.get_seed_nodes(G_ppi, node_scoring)
            expanded_nodes = self.HomoGraph.get_expanded_nodes(
                seed_nodes, all_proteins, pr_scores
            )
            nx_subgraph, non_isolated_nodes, node_scoring_filtered, node_index_filtered = (
                self.HomoGraph.filtering_info(G_ppi, expanded_nodes, node_scoring)
            )

            labels = generate_expanded_labels(
                seed_nodes,
                node_index_filtered,
                non_isolated_nodes
            )

            g = self.HomoGraph.convert_networkx_to_dgl_graph(
                nx_subgraph.subgraph(non_isolated_nodes).copy(),
                node_scoring_filtered,
                node_index_filtered,
                non_isolated_nodes,
                pr_scores
            )

            train_idx, val_idx, test_idx = split_train_test_val_indices(
                labels
            )

            os.makedirs(f'{self.output_path}/{disease}', exist_ok=True)
            best_trial = run_optuna_tuning(
                g,
                g.ndata['feat'],
                labels,
                train_idx,
                val_idx,
                disease,
                f'{self.output_path}/{disease}'
            )

            # Prepare model, optimizer and loss
            model, optimizer, loss_fn = builder(
                g, best_trial.params, labels, train_idx, self.device
            )

            # Training loop
            losses, _, val_f1s, val_precisions, val_recalls, best_threshold = training_loop(
                model, g, labels, train_idx, val_idx, optimizer, loss_fn, self.epochs, self.device
            )

            plot_loss_and_metrics(
                losses, val_f1s, val_recalls, val_precisions,
                save_path=f"{self.output_path}/{disease}"
                "/training_progress.png"
            )

            # Evaluation
            logits, test_acc, test_f1, test_prec, test_rec, test_auc, preds, _ = evaluating_model(
                model, g, labels, test_idx, self.device, threshold=best_threshold
            )
            displaying_metrics(test_acc, test_f1, test_rec, test_prec, test_auc, preds, test_idx)

            y_true, y_pred, y_scores = obtaining_pred_scores(
                labels, preds, logits, test_idx, self.device
            )
            plot_confusion_matrix(
                y_true, y_pred, save_path=f"{self.output_path}/{disease}"
                "/confusion_matrix.png"
            )

            precision, recall = plot_precision_recall_curve(
                y_true, y_scores,
                save_path=f"{self.output_path}/{disease}/pr_curve.png"
            )

            fpr, tpr = plot_roc_curve(
                y_true, y_scores,
                save_path=f"{self.output_path}/{disease}/roc_curve.png"
            )

            self.all_pr_curves.append((disease, precision, recall))
            self.all_roc_curves.append((disease, fpr, tpr))

            predicted_proteins = obtaining_predicted_proteins(
                node_index_filtered, preds, test_idx, seed_nodes
            )

            saving_results(
                predicted_proteins, seed_nodes, best_trial.params, y_true, y_scores,
                preds, model, test_prec, test_rec, test_f1, test_auc, test_acc,
                best_threshold, labels, test_idx, node_index_filtered,
                non_isolated_nodes, disease, self.output_path
            )

        plot_all_curves(
            self.all_pr_curves,
            curve_type="pr",
            save_path=f"{self.output_path}/all_diseases_pr_curve.png"
        )
        plot_all_curves(
            self.all_roc_curves,
            curve_type="roc",
            save_path=f"{self.output_path}/all_diseases_roc_curve.png"
        )
