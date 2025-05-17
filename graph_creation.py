from typing import Union

import dgl
import networkx as nx
import pandas as pd
from networkx import Graph
from sklearn.preprocessing import LabelEncoder


class GraphPPI():
    def __init__(self):
        pass

    def create_homogeneous_graph(self, df_pro_pro: pd.DataFrame) -> Graph:
        """
        Create a graph from the PPI data

        Args:
            df_pro_pro: dataframe that contains the protein-protein interaction

        Returns:
            Graph: graph created from this PPI
        """
        G_ppi = nx.from_pandas_edgelist(df_pro_pro, 'prA', 'prB')
        print(f"PPI Network: {G_ppi.number_of_nodes()} nodes, {G_ppi.number_of_edges()} edges")
        return G_ppi

    def create_heterogeneous_graph(self, df_dis_pro, df_pro_pro):
        # Encode proteins
        protein_encoder = LabelEncoder()
        all_proteins = pd.concat([df_pro_pro['prA'], df_pro_pro['prB'], df_dis_pro['protein_id']])
        protein_encoder.fit(all_proteins)
        df_pro_pro['src_id'] = protein_encoder.transform(df_pro_pro['prA'])
        df_pro_pro['dst_id'] = protein_encoder.transform(df_pro_pro['prB'])
        df_dis_pro['protein_id_enc'] = protein_encoder.transform(df_dis_pro['protein_id'])

        # Encode diseases
        disease_encoder = LabelEncoder()
        df_dis_pro['disease_id'] = disease_encoder.fit_transform(df_dis_pro['disease_name'])

        # 2. Build heterograph dictionary
        data_dict = {
            ('protein', 'interacts', 'protein'): (
                df_pro_pro['src_id'].values, df_pro_pro['dst_id'].values
            ),
            ('disease', 'associates', 'protein'): (
                df_dis_pro['disease_id'].values, df_dis_pro['protein_id_enc'].values
            ),
            ('protein', 'rev_associates', 'disease'): (
                df_dis_pro['protein_id_enc'].values, df_dis_pro['disease_id'].values
            )
        }

        # 3. Create heterograph
        hetero_graph = dgl.heterograph(data_dict)

        # Optional: print graph statistics
        print(hetero_graph)
        return hetero_graph

    def map_dis_gen(self, df_dis_pro: pd.DataFrame) -> dict:
        """
        Map the disease to the proteins that are associated with it

        Args:
            df_dis_pro: dataframe containing information about
            the interaction between proteins and diseases

        Returns:
            dict: dictionary where the keys are the diseases of
            interest and the value for each key (disease) is the
            set of proteins that are present in that disease
        """
        disease_pro_mapping = df_dis_pro.groupby("disease_name")["protein_id"].apply(set).to_dict()
        return disease_pro_mapping

    def main(self, df_pro_pro: pd.DataFrame, df_dis_pro: pd.DataFrame) -> Union[Graph, dict]:
        """
        Main function to create the graph and map the disease to the proteins

        Args:
            df_pro_pro: dataframe that contains the protein-protein interaction
            df_dis_pro: dataframe containing information about

        Returns:
            Graph: graph created from this PPI
            dict: dictionary where the keys are the diseases of
            interest and the value for each key (disease) is the
            set of proteins that are present in that disease
        """
        G_ppi = self.create_homogeneous_graph(df_pro_pro)
        G_dispro = self.create_heterogeneous_graph(df_dis_pro, df_pro_pro)
        disease_pro_mapping = self.map_dis_gen(df_dis_pro)
        return G_dispro, G_ppi, disease_pro_mapping
