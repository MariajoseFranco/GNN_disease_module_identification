import ast
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib_venn import venn2
from upsetplot import from_contents, plot

from utils import load_config


def config_setup():
    config = load_config()
    output_link_path = config['results_linkpred_dir']
    output_subg_path = config['results_subg_dir']
    disease_path = f"{output_link_path}/diseases_of_interest.csv"
    return disease_path, output_link_path, output_subg_path


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
    output = pd.read_csv(f'{output_link_path}/predicted_dis_pro.txt', sep='\t')
    for disease in diseases:
        results[disease]["Link Prediction"] = sorted(
            list(output[output['disease'] == disease]['protein'])
        )
    return results


def plot_upset(outputs, diseases):
    for disease in diseases:
        disease_output = outputs[disease]
        data = from_contents(disease_output)
        plot(data)
        plt.title(f"Overlap of Predicted Modules - Disease {disease}")
        plt.show()


def plot_venn_diagram(outputs, diseases):
    for disease in diseases:
        disease_output = outputs[disease]
        venn2(
            [
                set(disease_output['Subgraph Node Classification']),
                set(disease_output['Link Prediction'])
            ],
            set_labels=('Subgraph Node Classification', 'Link Prediction')
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
        venn2([set1, set2], set_labels=('Node Class.', 'Link Pred.'), ax=ax)
        ax.set_title(disease, fontsize=10)

    # Hide unused subplots
    for ax in axes[len(diseases):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_top_venn_intersections(disease_modules_dict, max_plots=6, ncols=3):
    """
    Plots Venn diagrams for the top N diseases with the largest intersection between two methods.
    """
    # Step 1: Compute intersection size for each disease
    disease_overlap = []
    for disease, methods in disease_modules_dict.items():
        set1 = set(methods.get('Subgraph Node Classification', []))
        set2 = set(methods.get('Link Prediction', []))
        intersection_size = len(set1 & set2)
        disease_overlap.append((disease, intersection_size, set1, set2))

    # Step 2: Sort diseases by intersection size (descending)
    disease_overlap.sort(key=lambda x: x[1], reverse=True)

    # Step 3: Select top N diseases
    top_diseases = disease_overlap[:max_plots]
    nrows = -(-max_plots // ncols)  # ceil division

    # Step 4: Plot
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*5))
    axes = axes.flatten() if max_plots > 1 else [axes]

    for i, (disease, intersection, set1, set2) in enumerate(top_diseases):
        venn2([set1, set2], set_labels=('Node Class.', 'Link Pred.'), ax=axes[i])
        axes[i].set_title(f"{disease}\nIntersection: {intersection}", fontsize=10)

    # Hide unused subplots
    for ax in axes[len(top_diseases):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_lowest_venn_intersections(disease_modules_dict, max_plots=6, ncols=3):
    """
    Plots Venn diagrams for the diseases with the lowest intersection between two methods.
    """
    # Step 1: Compute intersection size for each disease
    disease_overlap = []
    for disease, methods in disease_modules_dict.items():
        set1 = set(methods.get('Subgraph Node Classification', []))
        set2 = set(methods.get('Link Prediction', []))
        intersection_size = len(set1 & set2)
        disease_overlap.append((disease, intersection_size, set1, set2))

    # Step 2: Sort diseases by intersection size (ascending)
    disease_overlap.sort(key=lambda x: x[1])  # LOWEST first

    # Step 3: Select bottom N diseases
    selected_diseases = disease_overlap[:max_plots]
    nrows = -(-max_plots // ncols)  # ceil division

    # Step 4: Plot
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*5))
    axes = axes.flatten() if max_plots > 1 else [axes]

    for i, (disease, intersection, set1, set2) in enumerate(selected_diseases):
        venn2([set1, set2], set_labels=('Node Class.', 'Link Pred.'), ax=axes[i])
        axes[i].set_title(f"{disease}\nIntersection: {intersection}", fontsize=10)

    # Hide unused subplots
    for ax in axes[len(selected_diseases):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def compute_jaccard_summary(disease_modules_dict):
    records = []

    for disease, methods in disease_modules_dict.items():
        # Extract all sets
        method_sets = {
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
        df_result.to_csv('jaccard_summary.csv', sep='\t', index=False)

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


def table_to_latex(df, columns_interest, caption='caption', label='tab:tab', format_columns=None):
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
                             label=label)

    # Save to file (optional)
    with open(f"{label[4:]}_table.tex", "w") as f:
        f.write(latex_code)


def main():
    # Reading and combining outputs
    disease_path, output_link_path, output_subg_path = config_setup()
    diseases = reading_diseases(disease_path)
    output_classical = read_classical_outputs(diseases)
    output_subg = read_subg_outputs(output_subg_path, diseases, output_classical)
    combined_outputs = read_link_pred_outputs(output_link_path, diseases, output_subg)

    # Obtaining Jaccard indexes
    jacard_summary = compute_jaccard_summary(combined_outputs)
    table_to_latex(
        jacard_summary,
        ['Disease', 'Size(lcc)', 'Size(diamond)', 'Size(domino)', 'Size(topas)',
         'Size(robust)', 'Size(gnn_nc)', 'Size(gnn_lp)'],
        "Module sizes predicted per method for selected diseases.",
        "tab:module_sizes",
        ['Disease', 'LCC', 'DIAMOND', 'DOMINO', 'TOPAS', 'ROBUST', 'GNN NC', 'GNN LP']
    )

    # plot_jaccard_vs_intersection(jacard_summary, 'domino', 'gnn_lp')
    # plot_upset(combined_outputs, diseases)
    # plot_venn_diagram(combined_outputs, diseases)
    # plot_venn_grid(combined_outputs, max_plots=10)
    # plot_top_venn_intersections(combined_outputs)
    # plot_lowest_venn_intersections(combined_outputs)
    # print(jacard_summary)


if __name__ == "__main__":
    main()
