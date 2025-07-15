# GNN for disease module identification
**By:** Mariajose Fraco Orozco

**Tutor:** Lucía Prieto Santamaría

This project provides two pipelines for disease module identification using Graph Neural Networks (GNNs):

## Tasks

### 1. Link Prediction Task

- Predict **disease–protein** or **disease–drug** associations using a heterogeneous graph.

### 2. Subgraph Classification Task

- Classify disease modules using subgraphs and homogeneous GNN encoders.

---

## Project Structure

```
GNN_DISEASE_MODULE_IDENTIFICATION/
|
├── LinkPrediction/
|   ├── utils/                 # Utility functions (building models, saving results, etc.)
|   ├── dot_predictor.py       # Dot product predictor for link prediction
|   ├── mlp_predictor.py        # MLP predictor for link prediction
|   ├── heterogeneous_graph.py # Heterogeneous graph construction
|   ├── hyperparameter_tunning.py # Hyperparameter optimization (Optuna)
|   ├── main.py                # Entry point for Link Prediction
|   ├── pipeline.py            # Disease–protein pipeline
|   ├── pipeline_full_graph.py # Disease–drug pipeline
|   └── trainer.py             # Training loop
|
├── SubgraphClassification/
|   ├── utils/                  # Utility functions for subgraph classification
|   ├── GNN_encoder.py         # Homogeneous GNN for subgraph classification
|   ├── homogeneous_graph.py   # Subgraph generation and handling
|   ├── main.py                # Entry point for Subgraph Classification
|   └── pipeline.py            # Subgraph classification pipeline
|
├── data_compilation.py         # Data loading and preparation
├── api_data.py                 # API data handling
├── visualizations.py           # Visualization utilities
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── README.md                   # Project description
```

---

## How to Run

### Link Prediction Task

Run the **Link Prediction task** by executing:

```bash
PYTHONPATH=. python LinkPrediction/main.py
```

You will be prompted:

```
Do you want to run the full graph pipeline? (y/n):
```

- **If you choose `y`:**\
  The **full graph pipeline** will run, performing **disease–drug link prediction** `(pipeline_full_graph.py)`.

- **If you choose `n`:**\
  The **disease–protein link prediction** pipeline will run `(pipeline.py)`.

---

### Subgraph Classification Task

Run the **Subgraph Classification task** by executing:

```bash
PYTHONPATH=. python SubgraphClassification/main.py
```

This will obtain disease modules per subgraph depending on the diseases of study.

---

## Requirements

Install the dependencies using:

```bash
pip install -r requirements.txt
```

**Main libraries:**

- `dgl`
- `torch`
- `optuna`
- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`

---

## Outputs

Results (metrics, plots, and predictions) are saved automatically in the `outputs/` directories for each task.

---

## Contact

For questions or collaborations, please contact the project owner.

