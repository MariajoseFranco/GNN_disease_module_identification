import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data_compilation import DataCompilation
from GNN import GNN
from graph_creation import GraphPPI
from utils import create_edge_labels, generate_data


class Main():
    def __init__(self, path):
        # Select the diseases to work with
        self.selected_diseases = ["Albinism", "Alcohol Use Disorder"]
        self.DC = DataCompilation(path, self.selected_diseases)
        self.GPPI = GraphPPI()

    def run_gnn(self, G_ppi, disease_pro_mapping, MIN_SEEDS=10):
        for disease, all_seeds in tqdm(disease_pro_mapping.items()):
            print(f"Processing: {disease} ({len(all_seeds)} raw seeds)")

            seed_nodes = [node for node in all_seeds if node in G_ppi]
            if len(seed_nodes) < MIN_SEEDS:
                print("Skipped — not enough seeds in PPI")
                continue

    def train_gnn(self, model, data, edge_pairs, labels, epochs=100, lr=0.01):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            preds = model(data.x, data.edge_index, edge_pairs)
            loss = F.binary_cross_entropy_with_logits(preds, labels)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    def main(self):
        # Classical Methods
        df_pro_pro, df_gen_pro, df_dis_gen, df_dis_pro = self.DC.main()
        G_ppi, disease_pro_mapping = self.GPPI.main(df_pro_pro, df_dis_pro)
        data, node_index = generate_data(G_ppi)
        edge_pairs, labels = create_edge_labels(data.edge_index, num_nodes=data.num_nodes)
        # Train/test split
        train_idx, test_idx = train_test_split(range(len(labels)), test_size=0.2, random_state=42)
        train_edge_pairs = edge_pairs[:, train_idx]
        train_labels = labels[train_idx]

        # Initialize and train model
        model = GNN(in_channels=data.num_node_features, hidden_channels=64)
        self.train_gnn(model, data, train_edge_pairs, train_labels)


if __name__ == "__main__":
    path = "./data/"
    # path = "/app/data/"
    Main(path).main()
