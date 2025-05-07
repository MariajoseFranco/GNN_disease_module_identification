from typing import Union

import networkx as nx
import pandas as pd
from networkx import Graph


class GraphPPI():
    def __init__(self):
        pass

    def create_graph(self, df_pro_pro: pd.DataFrame) -> Graph:
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
        G_ppi = self.create_graph(df_pro_pro)
        disease_pro_mapping = self.map_dis_gen(df_dis_pro)
        return G_ppi, disease_pro_mapping
