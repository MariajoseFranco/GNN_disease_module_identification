import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DataCompilation():
    def __init__(self, data_path, disease_path) -> None:
        self.data_path = data_path
        self.disease_path = disease_path

    def get_matched_diseases(self, df_dis_pro, output_path):
        """
        Load and clean the disease names from a CSV file.

        The CSV is expected to have one column without a header.
        A new column 'disease_cleaned' is added with preprocessed disease names.

        Returns:
            pd.DataFrame: DataFrame with columns ['DISEASE', 'disease_cleaned'].
        """
        diseases = pd.read_csv(self.disease_path)
        cui_list = diseases['cui'].to_list()
        df_dis_pro_matched = df_dis_pro[df_dis_pro['cui'].isin(cui_list)]
        df_dis_pro_matched['disease_name'].to_csv(
            f'{output_path}/diseases_of_interest.csv', index=False
        )
        selected_diseases = df_dis_pro_matched['disease_name'].unique().tolist()
        # BORRAR lo de abajo
        # selected_diseases = selected_diseases[:3]
        # df_dis_pro_matched = df_dis_pro_matched[df_dis_pro_matched['disease_name'].isin(
        #     selected_diseases
        # )]
        return df_dis_pro_matched, selected_diseases

    def get_data(self):
        """
        Load interaction datasets from files:
            - Protein-Protein Interactions (PPI)
            - Gene-Protein Interactions
            - Disease-Gene Interactions

        The PPI dataset is filtered to remove self-loops (interactions where prA == prB).

        Returns:
            tuple: DataFrames (df_pro_pro, df_gen_pro, df_dis_gen)
        """
        # Protein - Protein Interaction
        df_pro_pro = pd.read_csv(f'{self.data_path}/pro_pro.tsv', sep='\t')
        df_pro_pro = df_pro_pro[df_pro_pro['prA'] != df_pro_pro['prB']]
        # Gen - Protein Interaction
        df_gen_pro = pd.read_csv(f'{self.data_path}/gen_pro.tsv', sep='\t')
        # Disease - Gen Interaction
        df_dis_gen = pd.read_csv(f'{self.data_path}/dis_gen.tsv', sep='\t')
        return df_pro_pro, df_gen_pro, df_dis_gen

    def get_dis_pro_data(self, df_dis_gen, df_gen_pro):
        """
        Merge the disease-gene and gene-protein datasets to create disease-protein interactions.
        Filters the data to include only the diseases matched with the selected list.

        Args:
            df_dis_gen (pd.DataFrame): Disease-Gene interaction data.
            df_gen_pro (pd.DataFrame): Gene-Protein interaction data.
            selected_diseases (pd.DataFrame): Preprocessed diseases to match.
            output_path (str): Path were the outputs are going to be stored.

        Returns:
            tuple:
                - pd.DataFrame: Filtered disease-protein interaction data.
                - list: List of matched disease names.
        """
        df_dis_pro = df_dis_gen.merge(
            df_gen_pro, how='left', on='gene_id', indicator=True
        )
        df_dis_pro = df_dis_pro[df_dis_pro['_merge'] == 'both'].drop(
            ['_merge'], axis=1
        )
        return df_dis_pro

    def encoding_diseases(self, df_dis_pro):
        """
        Encode disease names into integer IDs using LabelEncoder.

        Args:
            df_dis_pro (pd.DataFrame): Disease-Protein interaction
            data with a 'disease_name' column.

        Returns:
            pd.DataFrame: DataFrame with a new column 'disease_id'.
        """
        disease_encoder = LabelEncoder()
        df_dis_pro['disease_id'] = disease_encoder.fit_transform(df_dis_pro['disease_name'])
        return df_dis_pro

    def encoding_proteins(self, df_pro_pro, df_dis_pro):
        """
        Encode protein identifiers into integer IDs using LabelEncoder.

        Args:
            df_pro_pro (pd.DataFrame): Protein-Protein interaction data with columns ['prA', 'prB'].
            df_dis_pro (pd.DataFrame): Disease-Protein interaction data with column 'protein_id'.

        Returns:
            tuple:
                - pd.DataFrame: Encoded protein-protein interaction data
                with 'src_id' and 'dst_id' columns.
                - pd.DataFrame: Encoded disease-protein interaction data
                with 'protein_id_enc' column.
        """
        protein_encoder = LabelEncoder()
        all_proteins = pd.concat([df_pro_pro['prA'], df_pro_pro['prB'], df_dis_pro['protein_id']])
        protein_encoder.fit(all_proteins)
        df_pro_pro['src_id'] = protein_encoder.transform(df_pro_pro['prA'])
        df_pro_pro['dst_id'] = protein_encoder.transform(df_pro_pro['prB'])
        df_dis_pro['protein_id_enc'] = protein_encoder.transform(df_dis_pro['protein_id'])
        return df_pro_pro, df_dis_pro

    def main(self):
        """
        Executes the data preparation pipeline:
            - Loads interaction data.
            - Merges disease-gene and gene-protein interactions into disease-protein associations.
            - Encodes disease and protein IDs.
            - Returns the final encoded datasets.

        Args:
            selected_diseases (pd.DataFrame): DataFrame containing the preprocessed
            selected diseases.
            output_path (str): Path were the outputs are going to be stored

        Returns:
            tuple:
                - df_pro_pro_encoded (pd.DataFrame): Encoded protein-protein interactions.
                - df_gen_pro (pd.DataFrame): Gene-Protein interactions.
                - df_dis_gen (pd.DataFrame): Disease-Gene interactions.
                - df_dis_pro_encoded (pd.DataFrame): Encoded disease-protein associations.
                - diseases_matched (list): List of matched disease names.
        """
        df_pro_pro, df_gen_pro, df_dis_gen = self.get_data()
        df_dis_pro = self.get_dis_pro_data(
            df_dis_gen, df_gen_pro
        )
        df_dis_pro_encoded = self.encoding_diseases(df_dis_pro)
        df_pro_pro_encoded, df_dis_pro_encoded = self.encoding_proteins(
            df_pro_pro, df_dis_pro_encoded
        )
        return df_pro_pro_encoded, df_gen_pro, df_dis_gen, df_dis_pro_encoded
