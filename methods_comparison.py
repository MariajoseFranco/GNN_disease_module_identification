import ast
import json
import os
from collections import defaultdict
from itertools import combinations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.patches import Patch
from matplotlib_venn import venn2, venn3
from sklearn.metrics import (auc, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from upsetplot import from_contents, plot

from data_compilation import DataCompilation
from SubgraphClassification.homogeneous_graph import HomogeneousGraph
from utils import load_config


def config_setup():
    config = load_config()
    output_link_path = config['results_linkpred_dir']
    output_subg_path = config['results_subg_dir']
    data_path = config['data_dir']
    disease_path = f"{output_link_path}/diseases_of_interest.csv"
    comparison_path = "./comparison_outputs"
    dis_path = config['disease_dir']
    return disease_path, output_link_path, output_subg_path, comparison_path, data_path, dis_path


def reading_diseases(diseases_path):
    diseases = pd.read_csv(diseases_path, sep='\t')['DISEASES OF INTEREST'].to_list()
    return diseases


def read_classical_outputs(diseases):
    output = pd.read_csv(
        '/Users/mariajosefranco/Desktop/Data Science - UPM/TFM/project/GITLAB/'
        'disease-module-identification/src/outputs/multi_disease_modules.csv',
        sep=',',
        converters={'nodes': ast.literal_eval}
    )
    results = {}
    for disease in diseases:
        for method in output['method'].unique():
            df = output[output['disease'] == disease]
            df_method = df[df['method'] == method]
            if method == 'seed_nodes':
                results[disease] = {
                    method: df_method['nodes'].item()
                }
            else:
                results[disease][method] = df_method['nodes'].item()
    return results


def read_subg_outputs(output_subg_path, diseases, results):
    for disease in diseases:
        output = pd.read_csv(f'{output_subg_path}/{disease}/predicted_proteins.txt', sep='\t')
        results[disease]["Subgraph Node Classification"] = sorted(
            list(output["Predicted Proteins"])
        )
    return results


def read_link_pred_outputs(output_link_path, diseases, results):
    output = pd.read_csv(f'{output_link_path}/disease-protein2/predicted_dis_pro.txt', sep='\t')
    for disease in diseases:
        results[disease]["Link Prediction"] = sorted(
            list(output[output['disease'] == disease]['protein'])
        )
    return results


def plot_upset(outputs, diseases, path):
    for disease in diseases:
        disease_output = outputs[disease]
        del disease_output["lcc"]
        data = from_contents(disease_output)
        plot(data)
        plt.title(f"Overlap of Predicted Modules - Disease {disease}")
        plt.savefig(f"{path}/upset_plots/upset_{disease}.png", bbox_inches='tight')


def plot_venn_diagram(outputs, diseases):
    for disease in diseases:
        disease_output = outputs[disease]
        venn3(
            [
                set(disease_output['seed_nodes']),
                set(disease_output['Subgraph Node Classification']),
                set(disease_output['Link Prediction'])
            ],
            set_labels=('Seed Nodes', 'Subgraph Node Classification', 'Link Prediction')
        )
        plt.title(f"Disease {disease}")
        plt.show()


def plot_venn_grid(disease_modules_dict, max_plots=None, ncols=3):
    diseases = list(disease_modules_dict.keys())
    if max_plots:
        diseases = diseases[:max_plots]

    n = len(diseases)
    nrows = -(-n // ncols)  # ceil division
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*5))

    # Flatten axes array for easy indexing
    axes = axes.flatten() if n > 1 else [axes]

    for idx, disease in enumerate(diseases):
        ax = axes[idx]
        methods = disease_modules_dict[disease]
        set1 = set(methods.get('Subgraph Node Classification', []))
        set2 = set(methods.get('Link Prediction', []))
        set3 = set(methods.get('seed_nodes', []))
        venn3([set1, set2, set3], set_labels=('Node Class.', 'Link Pred.', 'Seeds'), ax=ax)
        ax.set_title(disease, fontsize=10)

    # Hide unused subplots
    for ax in axes[len(diseases):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_venn_grid_for_disease(disease_name, disease_modules_dict):
    methods = ['lcc', 'diamond', 'domino', 'topas', 'robust', 
               'Subgraph Node Classification', 'Link Prediction']

    method_labels = {
        'lcc': 'LCC',
        'diamond': 'DIAMOND',
        'domino': 'DOMINO',
        'topas': 'TOPAS',
        'robust': 'ROBUST',
        'Subgraph Node Classification': 'GNN NC',
        'Link Prediction': 'GNN LP'
    }

    disease_data = disease_modules_dict.get(disease_name)
    if not disease_data:
        print(f"Disease '{disease_name}' not found.")
        return

    seed_nodes = set(disease_data.get('seed_nodes', []))

    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.flatten()

    for i, method in enumerate(methods):
        ax = axs[i]
        method_nodes = set(disease_data.get(method, []))

        v = venn2([seed_nodes, method_nodes],
                  set_labels=('', ''),  # Remove labels under the sets
                #   set_colors=('#ff0000', '#0000ff'),
                  set_colors=('#ff6666', '#66cc66'),
                  ax=ax)

        ax.set_title(f'{method_labels[method]} vs Seed Nodes')

        handles = []
        # Unique to seed nodes
        if v.get_patch_by_id('10'):
            handles.append(Patch(facecolor=v.get_patch_by_id('10').get_facecolor(), label='Seed Nodes'))
        # Unique to method
        if v.get_patch_by_id('01'):
            handles.append(Patch(facecolor=v.get_patch_by_id('01').get_facecolor(), label=f'{method_labels[method]}'))
        # Intersection
        if v.get_patch_by_id('11'):
            handles.append(Patch(facecolor=v.get_patch_by_id('11').get_facecolor(), label='Intersection'))

        ax.legend(handles=handles, loc='upper right', fontsize=8, frameon=False)

    axs[-1].axis('off')
    fig.suptitle(f"Venn Diagrams: {disease_name}", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_top_venn_intersections(disease_modules_dict, path, max_plots=6, ncols=3):
    """
    Plots Venn diagrams for the top N diseases with the largest intersection between two methods.
    """
    # Step 1: Compute intersection size for each disease
    disease_overlap = []
    for disease, methods in disease_modules_dict.items():
        if disease in ['Colorectal Carcinoma', 'Diabetes Mellitus, Non-Insulin-Dependent', 'Rheumatoid Arthritis']:
            set1 = set(methods.get('Subgraph Node Classification', []))
            set2 = set(methods.get('Link Prediction', []))
            set3 = set(methods.get('seed_nodes', []))
            intersection_size = len(set1 & set2 & set3)
            disease_overlap.append((disease, intersection_size, set1, set2, set3))

    # Step 2: Sort diseases by intersection size (descending)
    disease_overlap.sort(key=lambda x: x[1], reverse=True)

    # Step 3: Select top N diseases
    top_diseases = disease_overlap[:max_plots]
    nrows = -(-max_plots // ncols)  # ceil division

    # Step 4: Plot
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*5))
    axes = axes.flatten() if max_plots > 1 else [axes]

    for i, (disease, intersection, set1, set2, set3) in enumerate(top_diseases):
        venn3([set1, set2, set3], set_labels=('Node Class.', 'Link Pred.', 'Seeds'), set_colors=('#ff6666', '#66cc66', '#6666cc'), ax=axes[i])
        axes[i].set_title(f"{disease}\nIntersection: {intersection}", fontsize=10)

    # Hide unused subplots
    for ax in axes[len(top_diseases):]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{path}/top3_venn_diagram.png", bbox_inches='tight')


def plot_lowest_venn_intersections(disease_modules_dict, path, max_plots=6, ncols=3):
    """
    Plots Venn diagrams for the diseases with the lowest intersection between two methods.
    """
    # Step 1: Compute intersection size for each disease
    disease_overlap = []
    for disease, methods in disease_modules_dict.items():
        if disease in ['Hyperpipecolic Acidemia', 'Panniculitis, Lupus Erythematosus', 'Cerebrovascular Occlusion']:
            set1 = set(methods.get('Subgraph Node Classification', []))
            set2 = set(methods.get('Link Prediction', []))
            set3 = set(methods.get('seed_nodes', []))
            intersection_size = len(set1 & set2 & set3)
            disease_overlap.append((disease, intersection_size, set1, set2, set3))

    # Step 2: Sort diseases by intersection size (ascending)
    disease_overlap.sort(key=lambda x: x[1])  # LOWEST first

    # Step 3: Select bottom N diseases
    selected_diseases = disease_overlap[:max_plots]
    nrows = -(-max_plots // ncols)  # ceil division

    # Step 4: Plot
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*5))
    axes = axes.flatten() if max_plots > 1 else [axes]

    for i, (disease, intersection, set1, set2, set3) in enumerate(selected_diseases):
        venn3([set1, set2, set3], set_labels=('Node Class.', 'Link Pred.', 'Seeds'), ax=axes[i])
        axes[i].set_title(f"{disease}\nIntersection: {intersection}", fontsize=10)

    # Hide unused subplots
    for ax in axes[len(selected_diseases):]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{path}/lowest3_venn_diagram.png", bbox_inches='tight')


def compute_jaccard_summary(disease_modules_dict, path):
    records = []

    for disease, methods in disease_modules_dict.items():
        # Extract all sets
        method_sets = {
            'seed_nodes': set(methods.get('seed_nodes', [])),
            'lcc': set(methods.get('lcc', [])),
            'diamond': set(methods.get('diamond', [])),
            'domino': set(methods.get('domino', [])),
            'topas': set(methods.get('topas', [])),
            'robust': set(methods.get('robust', [])),
            'gnn_nc': set(methods.get('Subgraph Node Classification', [])),
            'gnn_lp': set(methods.get('Link Prediction', []))
        }

        # Compute Jaccard indices for all method pairs
        pairwise_jaccards = {}
        for (m1, s1), (m2, s2) in combinations(method_sets.items(), 2):
            key = f'J({m1}, {m2})'
            pairwise_jaccards[key] = jaccard(s1, s2)

        # Also compute individual module sizes
        sizes = {f'Size({method})': len(nodes) for method, nodes in method_sets.items()}

        record = {'Disease': disease}
        record.update(pairwise_jaccards)
        record.update(sizes)

        records.append(record)

        df_result = pd.DataFrame(records).sort_values(by='Disease')
        df_result.to_csv(f'{path}/jaccard_summary.csv', sep='\t', index=False)

    return df_result


def jaccard(set1, set2):
    union = set1 | set2
    intersection = set1 & set2
    return len(intersection) / len(union) if union else 0.0


def plot_jaccard_vs_intersection(df_summary, method1='gnn_nc', method2='gnn_lp', top_n=10):
    """
    Plots Jaccard index vs. estimated intersection size for a given method pair across diseases.

    Parameters:
        df_summary: DataFrame with columns like 'J(method1, method2)',
        'Size(method1)', 'Size(method2)', and 'Disease'
        method1: str - name of first method (must match name used in compute_jaccard_summary)
        method2: str - name of second method
        top_n: int - number of diseases to display with highest estimated intersection
    """
    jacc_col = f'J({method1}, {method2})'
    size1_col = f'Size({method1})'
    size2_col = f'Size({method2})'

    # Compute estimated intersection size from Jaccard index
    df_summary['Estimated Intersection'] = df_summary.apply(
        lambda row: row[jacc_col] * (row[size1_col] + row[size2_col]) /
        (1 + row[jacc_col]) if row[jacc_col] > 0 else 0,
        axis=1
    )

    # Select top N by estimated intersection size
    top_df = df_summary.sort_values(by='Estimated Intersection', ascending=False).head(top_n)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=top_df,
        x='Estimated Intersection',
        y=jacc_col,
        hue='Disease',
        palette='tab10',
        s=80
    )

    plt.title(
        f'Top {top_n} Diseases - Jaccard Index vs Estimated '
        f'Intersection Size\n({method1} vs {method2})'
    )
    plt.xlabel(f'Estimated Intersection Size ({method1} ∩ {method2})')
    plt.ylabel(f'Jaccard Index ({method1} vs {method2})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()


def table_to_latex(
        df, path, columns_interest, caption='caption', label='tab:tab', format_columns=None,
):
    # Keep only the columns of interest
    df = df[columns_interest]

    # Optional: Rename columns for LaTeX clarity
    if format_columns:
        df.columns = format_columns

    # Generate LaTeX code
    latex_code = df.to_latex(index=False,
                             longtable=False,
                             escape=False,
                             caption=caption,
                             label=label,
                             float_format="%.3f")

    # Save to file (optional)
    with open(f"{path}/{label[4:]}_table.tex", "w") as f:
        f.write(latex_code)
    return df


def find_best_f1_result(path, diseases):
    best_f1 = -1
    best_file = None
    best_data = None

    for disease in diseases:
        json_file = f"{path}/{disease}/model/evaluation_metrics.json"
        with open(json_file) as f:
            data = json.load(f)
            f1 = data.get("test_f1", -1)
            if f1 > best_f1:
                best_f1 = f1
                best_file = json_file
                best_data = data

    return best_file, best_data


def obtaining_macro_averaged_metrics(diseases, path):
    f1s = []
    precisions = []
    recalls = []
    aucs = []
    accs = []

    for disease in diseases:
        y_true = torch.load(f"{path}/{disease}/model/y_true.pt")
        y_pred = torch.load(f"{path}/{disease}/model/preds.pt")
        y_scores = torch.load(f"{path}/{disease}/model/y_scores.pt")

        f1s.append(f1_score(y_true, y_pred))
        precisions.append(precision_score(y_true, y_pred))
        recalls.append(recall_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_scores))
        accs.append((y_pred == y_true).float().mean().item())

    # Macro averages
    macro_f1 = np.mean(f1s)
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_auc = np.mean(aucs)
    macro_acc = np.mean(accs)
    metrics_dict = {
        "mean_averaged_precision": float(macro_precision),
        "mean_averaged_recall": float(macro_recall),
        "mean_averaged_f1": float(macro_f1),
        "mean_averaged_auc": float(macro_auc),
        "mean_averaged_accuracy": float(macro_acc)
    }
    with open(f"{path}/mean_averaged_evaluation_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)


def plot_macro_averaged_pr_curve(diseases, path_subg, path_lp, path):
    precisions_list = []

    for disease in diseases:
        y_true = torch.load(f"{path_subg}/{disease}/model/y_true.pt").numpy()
        y_scores = torch.load(f"{path_subg}/{disease}/model/y_scores.pt").numpy()

        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        recall_interp = np.linspace(0, 1, 100)
        precision_interp = np.interp(recall_interp, recall[::-1], precision[::-1])
        precisions_list.append(precision_interp)

    mean_precision = np.mean(precisions_list, axis=0)
    mean_recall = recall_interp
    mean_auc = auc(mean_recall, mean_precision)

    # Link Prediction
    y_true_lp = torch.load(f"{path_lp}/disease-protein2/model/y_true.pt").numpy()
    y_scores_lp = torch.load(f"{path_lp}/disease-protein2/model/y_scores.pt").numpy()
    precision_lp, recall_lp, _ = precision_recall_curve(y_true_lp, y_scores_lp)
    auc_lp = auc(recall_lp, precision_lp)

    plt.figure(figsize=(7, 5))
    plt.plot(mean_recall, mean_precision, label=f"Subgraph Classification (AUC={mean_auc:.2f})", linewidth=2)
    plt.plot(recall_lp, precision_lp, label=f"Link Prediction (AUC={auc_lp:.2f})", linestyle='--', linewidth=2)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(path, exist_ok=True)
    plt.savefig(f"{path}/comparison_pr_curve.png", dpi=300)


def plot_macro_averaged_roc_curve(diseases, path_subg, path_lp, path):
    tpr_list = []

    for disease in diseases:
        try:
            y_true = torch.load(f"{path_subg}/{disease}/model/y_true.pt").numpy()
            y_scores = torch.load(f"{path_subg}/{disease}/model/y_scores.pt").numpy()

            # Skip if less than 2 classes are present
            if len(np.unique(y_true)) < 2:
                print(f"Skipping {disease}: only one class present in y_true.")
                continue

            fpr, tpr, _ = roc_curve(y_true, y_scores)
            fpr_interp = np.linspace(0, 1, 100)
            tpr_interp = np.interp(fpr_interp, fpr, tpr)
            tpr_list.append(tpr_interp)

        except Exception as e:
            print(f"Skipping {disease} due to error: {e}")
    if len(tpr_list) == 0:
        print("No valid ROC curves found for Subgraph Node Classification.")
        return

    mean_tpr = np.mean(tpr_list, axis=0)
    mean_fpr = fpr_interp
    mean_auc = auc(mean_fpr, mean_tpr)

    # Link Prediction
    y_true_lp = torch.load(f"{path_lp}/disease-protein2/model/y_true.pt").numpy()
    y_scores_lp = torch.load(f"{path_lp}/disease-protein2/model/y_scores.pt").numpy()
    fpr_lp, tpr_lp, _ = roc_curve(y_true_lp, y_scores_lp)
    auc_lp = auc(fpr_lp, tpr_lp)

    # Plot
    plt.figure(figsize=(7, 5))
    plt.plot(mean_fpr, mean_tpr, label=f"Subgraph Classification (AUC={mean_auc:.2f}, N={len(tpr_list)})", linewidth=2)
    plt.plot(fpr_lp, tpr_lp, label=f"Link Prediction (AUC={auc_lp:.2f})", linestyle='--', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(path, exist_ok=True)
    plt.savefig(f"{path}/comparison_roc_curve.png", dpi=300)


def plot_aggregated_confusion_matrix(diseases, path_subg, save_path):
    all_preds = []
    all_labels = []

    for disease in diseases:
        try:
            y_true = torch.load(f"{path_subg}/{disease}/model/y_true.pt").numpy()
            y_pred = torch.load(f"{path_subg}/{disease}/model/preds.pt").numpy()

            # Skip if only one class is present in y_true
            if len(np.unique(y_true)) < 2:
                print(f"Skipping {disease} due to only one class in y_true.")
                continue

            all_labels.append(y_true)
            all_preds.append(y_pred)

        except Exception as e:
            print(f"Skipping {disease} due to error: {e}")

    if len(all_preds) == 0:
        print("No valid predictions found.")
        return

    # Concatenate across diseases
    y_true_total = np.concatenate(all_labels)
    y_pred_total = np.concatenate(all_preds)

    # Compute confusion matrix
    cm = confusion_matrix(y_true_total, y_pred_total)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Aggregated Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{save_path}/confusion_matrix_aggregated.png", dpi=300)
    plt.show()


def evaluate_classical_on_ppi(all_proteins, predicted_proteins, seed_proteins):
    """
    Evaluate classical method over the full PPI protein space.

    Args:
        all_proteins (set[str]): All protein names in the PPI network.
        predicted_proteins (list[str]): List of predicted positive proteins.
        seed_proteins (set[str]): Ground-truth positives (known disease proteins).

    Returns:
        dict: Precision, recall, and F1-score.
    """
    y_true = [1 if p in seed_proteins else 0 for p in all_proteins]
    y_pred = [1 if p in predicted_proteins else 0 for p in all_proteins]

    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }


def classical_methods_eval(diseases, method_predictions):
    all_nodes = pd.read_csv("/Users/mariajosefranco/Desktop/Data Science - UPM/TFM/project/GNN_disease_module_identification/comparison_outputs/ppi_nodes.csv")["Protein Nodes"].tolist()
    results = {}
    for disease in diseases:
        disease_dict = method_predictions[disease]
        seeds = set(disease_dict['seed_nodes'])

        results[disease] = {}

        for method in ['robust', 'domino', 'lcc', 'topas']:
            preds = disease_dict[method]
            metrics = evaluate_classical_on_ppi(
                all_proteins=all_nodes,
                predicted_proteins=preds,
                seed_proteins=seeds
            )
            results[disease][method] = metrics

    with open("classical_method_evaluation.json", "w") as f:
        json.dump(results, f, indent=4)

    methods = ['robust', 'domino', 'lcc', 'topas']
    acc = {method: defaultdict(list) for method in methods}

    # Accumulate metrics
    for disease_metrics in results.values():
        for method in methods:
            if method in disease_metrics:
                for metric in ['precision', 'recall', 'f1']:
                    acc[method][metric].append(disease_metrics[method][metric])

    # Compute mean metrics per method
    summary = {
        method: {
            metric: sum(values)/len(values) if values else None
            for metric, values in metrics.items()
        }
        for method, metrics in acc.items()
    }

    df = pd.DataFrame(summary).T
    df = df[['precision', 'recall', 'f1']]
    print(df)
    return results


def avg_jaccard(df_jaccard):
    average_similarities = df_jaccard.drop(columns=['Disease']).mean().sort_values(ascending=False)

    # Convert to DataFrame for better display
    avg_sim_df = average_similarities.reset_index()
    avg_sim_df.columns = ['Method Pair', 'Average Jaccard Similarity']
    return avg_sim_df


def extract_extremes(row):
    max_col = row[1:].astype(float).idxmax()
    min_col = row[1:].astype(float).idxmin()
    return pd.Series({
        "Disease": row["Disease"],
        "Most Similar Pair": max_col,
        "Max Jaccard": row[max_col],
        "Least Similar Pair": min_col,
        "Min Jaccard": row[min_col],
    })


def jaccard_heatmap(df_jaccard, path):
    method_pairs = df_jaccard.columns[1:]  # Exclude the 'Disease' column
    avg_jaccard_dict = df_jaccard[method_pairs].mean().to_dict()

    # Extract unique methods from method pairs
    import re
    method_set = set()
    for pair in method_pairs:
        matches = re.findall(r'J\(([^,]+), ([^)]+)\)', pair)
        if matches:
            a, b = matches[0]
            method_set.update([a.strip(), b.strip()])
    methods = sorted(method_set)

    # Initialize the matrix
    heatmap_matrix = pd.DataFrame(np.nan, index=methods, columns=methods)

    # Fill matrix symmetrically
    for pair, val in avg_jaccard_dict.items():
        matches = re.findall(r'J\(([^,]+), ([^)]+)\)', pair)
        if matches:
            a, b = matches[0]
            a, b = a.strip(), b.strip()
            heatmap_matrix.loc[a, b] = val
            heatmap_matrix.loc[b, a] = val

    # Fill diagonal with 1.0 (Jaccard similarity with self)
    np.fill_diagonal(heatmap_matrix.values, 1.0)

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_matrix, annot=True, fmt=".2f", cmap="YlGnBu", square=True, linewidths=0.5)
    plt.title("Average Jaccard Similarity Between Methods")
    plt.tight_layout()
    plt.savefig(f"{path}/jaccard_heatmap.png", dpi=300)


def visualize_modules(graph_dict, seed_nodes, G, disease, method):
    # Build full subgraph with all nodes in all modules
    all_nodes = set()
    for k in graph_dict:
        all_nodes.update(graph_dict[k])
    subG = G.subgraph(all_nodes).copy()

    if len(subG.nodes) == 0:
        print("Nothing to visualize: combined subgraph is empty.")
        return

    # Assign a unique color to each module
    color_map = {}
    module_keys = [k for k in graph_dict if k != "seed_nodes"]
    color_list = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    color_list = [c for c in color_list if c not in ['red', 'orange']]  # reserve red/orange

    for i, key in enumerate(module_keys):
        color_map[key] = color_list[i % len(color_list)]

    # Assign colors per node (seeds in red override any module color)
    node_colors = {}
    for key, color in color_map.items():
        for node in graph_dict[key]:
            if node not in seed_nodes:
                node_colors[node] = color

    for seed in seed_nodes:
        if seed in subG:
            node_colors[seed] = 'red'  # override with red

    # Set layout
    pos = nx.spring_layout(subG, seed=42, k=0.5)
    plt.figure(figsize=(12, 10))

    # Conditional drawing order based on method
    if method.lower() != "lcc":
        # Draw modules first, seed nodes after (seed nodes don't override)
        for key, color in color_map.items():
            module_nodes = set(graph_dict[key]) & set(subG.nodes)
            nx.draw_networkx_nodes(
                subG, pos,
                nodelist=list(module_nodes),
                node_color=color,
                label=key,
                node_size=300,
                alpha=0.8
            )

        # Draw seed nodes on top
        seeds_in_subG = list(seed_nodes & subG.nodes)
        nx.draw_networkx_nodes(
            subG, pos,
            nodelist=seeds_in_subG,
            node_color='red',
            label='seed_nodes',
            node_size=300
        )
    else:
        # Draw seed nodes first (so modules will override red)
        seeds_in_subG = list(seed_nodes & subG.nodes)
        nx.draw_networkx_nodes(
            subG, pos,
            nodelist=seeds_in_subG,
            node_color='red',
            label='seed_nodes',
            node_size=300
        )

        for key, color in color_map.items():
            module_nodes = set(graph_dict[key])
            nx.draw_networkx_nodes(
                subG, pos,
                nodelist=list(module_nodes),
                node_color=color,
                label=key,
                node_size=300,
                alpha=0.8
            )

    # Edges and labels
    nx.draw_networkx_edges(subG, pos, alpha=0.3)
    nx.draw_networkx_labels(subG, pos, font_size=7)

    plt.title(f"{method.upper()} Modules - {disease.title()}")
    plt.axis('off')
    plt.legend()
    plt.savefig(f"/Users/mariajosefranco/Desktop/Data Science - UPM/TFM/project/GNN_disease_module_identification/comparison_outputs/{disease}_{method}_graph.png")


def confusion_matrix_per_disease_linkpred(df, save_path):
    target_disease = 'Colorectal Carcinoma'
    df_sub = df[df['disease'] == target_disease]

    # Calcular matriz de confusión
    cm = confusion_matrix(df_sub['true'], df_sub['pred'])

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{save_path}/confusion_matrix_colorectal_carcinoma_lp.png", dpi=300)
    plt.show()


def build_prediction_dataframe(preds, y_true, test_pos_u, test_pos_v, test_neg_u, test_neg_v,
                               diseases_to_encoded_ids, proteins_to_encoded_ids):
    """
    Reconstruye el DataFrame completo con predicciones, etiquetas reales y enfermedad por muestra.
    """
    test_pos_u = torch.tensor(test_pos_u)
    test_pos_v = torch.tensor(test_pos_v)
    test_neg_u = torch.tensor(test_neg_u)
    test_neg_v = torch.tensor(test_neg_v)

    # Concatenar nodos enfermedad/proteína
    all_u = torch.cat([test_pos_u, test_neg_u])
    all_v = torch.cat([test_pos_v, test_neg_v])

    # Mapear a nombres
    diseases = [diseases_to_encoded_ids[u.item()] for u in all_u]
    proteins = [proteins_to_encoded_ids[v.item()] for v in all_v]

    # Crear DataFrame completo
    df = pd.DataFrame({
        'disease': diseases,
        'protein': proteins,
        'true': y_true,
        'pred': preds
    })

    return df


def read_vectors_linkpred(path):
    import pickle
    y_true = torch.load(f'{path}/disease-protein2/model/y_true.pt').numpy()
    preds = torch.load(f'{path}/disease-protein2/model/preds.pt').numpy()
    test_pos_u = torch.load(f'{path}/disease-protein2/model/test_pos_u.pt').numpy()
    test_pos_v = torch.load(f'{path}/disease-protein2/model/test_pos_v.pt').numpy()
    test_neg_u = torch.load(f'{path}/disease-protein2/model/test_neg_u.pt').numpy()
    test_neg_v = torch.load(f'{path}/disease-protein2/model/test_neg_v.pt').numpy()

    with open(f'{path}/disease-protein2/model/diseases_to_encoded_ids.pkl', 'rb') as f:
        diseases_to_encoded_ids = pickle.load(f)

    # Leer proteins_to_encoded_ids
    with open(f'{path}/disease-protein2/model/proteins_to_encoded_ids.pkl', 'rb') as f:
        proteins_to_encoded_ids = pickle.load(f)

    return y_true, preds, test_pos_u, test_pos_v, test_neg_u, test_neg_v, diseases_to_encoded_ids, proteins_to_encoded_ids


def main():
    # Reading and combining outputs
    disease_path, output_link_path, output_subg_path, comparison_path, data_path, dis_path = config_setup()
    os.makedirs(comparison_path, exist_ok=True)
    diseases = reading_diseases(disease_path)
    output_classical = read_classical_outputs(diseases)
    output_subg = read_subg_outputs(output_subg_path, diseases, output_classical)
    combined_outputs = read_link_pred_outputs(output_link_path, diseases, output_subg)

    # DC = DataCompilation(data_path, dis_path, comparison_path)
    # HomoGraph = HomogeneousGraph()
    # df_pro_pro, _, _, _, _ = DC.main()
    # G_ppi = HomoGraph.create_graph(df_pro_pro)
    # disease = 'Exophthalmos'
    # gout_dict = combined_outputs[disease]
    # for method in ["Subgraph Node Classification", "Link Prediction"]:
    #     visualize_modules(
    #         {method: gout_dict[method]}, gout_dict["seed_nodes"], G_ppi, disease, method
    #     )

    # JACCARD CALCULATIONS
    # jacard_summary = compute_jaccard_summary(combined_outputs, comparison_path)
    # jaccard_table = table_to_latex(
    #     jacard_summary,
    #     comparison_path,
    #     ['Disease', 'J(seed_nodes, lcc)', 'J(seed_nodes, diamond)',
    #      'J(seed_nodes, domino)', 'J(seed_nodes, topas)',
    #      'J(seed_nodes, robust)', 'J(seed_nodes, gnn_nc)',
    #      'J(seed_nodes, gnn_lp)', 'J(lcc, diamond)', 'J(lcc, domino)',
    #      'J(lcc, topas)', 'J(lcc, robust)', 'J(lcc, gnn_nc)', 'J(lcc, gnn_lp)',
    #      'J(diamond, domino)', 'J(diamond, topas)', 'J(diamond, robust)',
    #      'J(diamond, gnn_nc)', 'J(diamond, gnn_lp)', 'J(domino, topas)',
    #      'J(domino, robust)', 'J(domino, gnn_nc)', 'J(domino, gnn_lp)',
    #      'J(topas, robust)', 'J(topas, gnn_nc)', 'J(topas, gnn_lp)',
    #      'J(robust, gnn_nc)', 'J(robust, gnn_lp)', 'J(gnn_nc, gnn_lp)'],
    #     "Jaccard Indexes per Method",
    #     "tab:jaccard",
    #     ['Disease', 'J(SEEDS, LCC)', 'J(SEEDS, DIAMOND)',
    #      'J(SEEDS, DOMINO)', 'J(SEEDS, TOPAS)',
    #      'J(SEEDS, ROBUST)', 'J(SEEDS, GNN NC)',
    #      'J(SEEDS, GNN LP)', 'J(LCC, DIAMOND)', 'J(LCC, DOMINO)',
    #      'J(LCC, TOPAS)', 'J(LCC, ROBUST)', 'J(LCC, GNN NC)', 'J(LCC, GNN LP)',
    #      'J(DIAMOND, DOMINO)', 'J(DIAMOND, TOPAS)', 'J(DIAMOND, ROBUST)',
    #      'J(DIAMOND, GNN NC)', 'J(DIAMOND, GNN LP)', 'J(DOMINO, TOPAS)',
    #      'J(DOMINO, ROBUST)', 'J(DOMINO, GNN NC)', 'J(DOMINO, GNN LP)',
    #      'J(TOPAS, ROBUST)', 'J(TOPAS, GNN NC)', 'J(TOPAS, GNN LP)',
    #      'J(ROBUST, GNN NC)', 'J(ROBUST, GNN LP)', 'J(GNN NC, GNN LP)']
    # )
    # df_avg_jaccard = avg_jaccard(jaccard_table)
    # table_to_latex(
    #     df_avg_jaccard,
    #     comparison_path,
    #     ['Method Pair', 'Average Jaccard Similarity'],
    #     "Average Jaccard Similarity",
    #     "tab:avg_jaccard"
    # )
    # # Apply to each row
    # jaccard_table_gnn = jaccard_table[['Disease', 'J(SEEDS, GNN NC)',
    #    'J(SEEDS, GNN LP)','J(LCC, GNN NC)', 'J(LCC, GNN LP)',
    #    'J(DIAMOND, GNN NC)', 'J(DIAMOND, GNN LP)', 'J(DOMINO, GNN NC)', 'J(DOMINO, GNN LP)',
    #    'J(TOPAS, GNN NC)', 'J(TOPAS, GNN LP)',
    #    'J(ROBUST, GNN NC)', 'J(ROBUST, GNN LP)', 'J(GNN NC, GNN LP)']]
    # extreme_similarities = jaccard_table_gnn.apply(extract_extremes, axis=1)

    # # Convert numerical results to float for clarity
    # extreme_similarities["Max Jaccard"] = extreme_similarities["Max Jaccard"].astype(float)
    # extreme_similarities["Min Jaccard"] = extreme_similarities["Min Jaccard"].astype(float)

    # table_to_latex(
    #     extreme_similarities,
    #     comparison_path,
    #     ['Disease', 'Most Similar Pair', 'Max Jaccard', 'Least Similar Pair', 'Min Jaccard'],
    #     "Maximum and Minimum Jaccard Similarity Between Pair of Methods",
    #     "tab:max_min_jaccard"
    # )

    # jaccard_heatmap(jaccard_table, comparison_path)

    # table_to_latex(
    #     jacard_summary,
    #     comparison_path,
    #     ['Disease', 'Size(lcc)', 'Size(diamond)', 'Size(domino)', 'Size(topas)',
    #      'Size(robust)', 'Size(gnn_nc)', 'Size(gnn_lp)'],
    #     "Module sizes predicted per method for selected diseases.",
    #     "tab:module_sizes",
    #     ['Disease', 'LCC', 'DIAMOND', 'DOMINO', 'TOPAS', 'ROBUST', 'GNN NC', 'GNN LP']
    # )

    # plot_jaccard_vs_intersection(jacard_summary, 'seed_nodes', 'gnn_lp')

    # UPSET PLOTS
    plot_upset(combined_outputs, diseases, comparison_path)

    # VEN DIAGRAMS PLOTS
    # plot_top_venn_intersections(combined_outputs, comparison_path, max_plots=3, ncols=3)
    # plot_lowest_venn_intersections(combined_outputs, comparison_path, max_plots=3, ncols=3)
    # plot_venn_grid_for_disease('Colorectal Carcinoma', combined_outputs)

    # MATRIZ DE CONFUSION LINK PRED
    # y_true, preds, test_pos_u, test_pos_v, test_neg_u, test_neg_v, diseases_to_encoded_ids, proteins_to_encoded_ids = read_vectors_linkpred(output_link_path)
    # df = build_prediction_dataframe(preds, y_true, test_pos_u, test_pos_v, test_neg_u, test_neg_v, diseases_to_encoded_ids, proteins_to_encoded_ids)
    # confusion_matrix_per_disease_linkpred(df, comparison_path)



    # results = classical_methods_eval(diseases, combined_outputs)

    # best_file, best_data = find_best_f1_result(output_subg_path, diseases)
    # print(f"Best F1: {best_data['test_f1']:.4f} from file {best_file}")


    # METRICAS SUBGRAPH CLASSIFICATION
    # obtaining_macro_averaged_metrics(diseases, output_subg_path)

    # PR Y ROC CURVES DE SUBG Y LINKPRED
    # plot_macro_averaged_pr_curve(diseases, output_subg_path, output_link_path, comparison_path)
    # plot_macro_averaged_roc_curve(diseases, output_subg_path, output_link_path, comparison_path)

    # MATRIZ DDE CONFUSION COMPLETA DE SUBG
    # plot_aggregated_confusion_matrix(diseases, output_subg_path, output_subg_path)


if __name__ == "__main__":
    main()
