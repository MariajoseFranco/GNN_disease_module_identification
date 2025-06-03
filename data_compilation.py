import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils import process_text


class DataCompilation():
    def __init__(self, data_path, disease_path) -> None:
        self.data_path = data_path
        self.disease_path = disease_path

    def get_diseases(self):
        """
        Load and clean the disease names from a CSV file.

        The CSV is expected to have one column without a header.
        A new column 'disease_cleaned' is added with preprocessed disease names.

        Returns:
            pd.DataFrame: DataFrame with columns ['DISEASE', 'disease_cleaned'].
        """
        diseases = pd.read_csv(self.disease_path, header=None)
        diseases = diseases.rename({0: 'DISEASE'}, axis=1)
        diseases['disease_cleaned'] = diseases['DISEASE'].apply(lambda x: process_text(x))
        return diseases

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

    def get_matched_diseases(self, selected_diseases, unique_diseases):
        """
        Match diseases from the selected list to unique disease names in the data
        by comparing preprocessed sets of words.

        Args:
            selected_diseases (pd.DataFrame): DataFrame with a 'disease_cleaned' column (as sets).
            unique_diseases (pd.DataFrame): DataFrame with a 'disease_name_cleaned'
            column (as sets).

        Returns:
            list: Matched disease names from the unique_diseases DataFrame.
        """
        selected_diseases['name_set'] = selected_diseases['disease_cleaned'].apply(set)
        unique_diseases['name_set'] = unique_diseases['disease_name_cleaned'].apply(set)

        matches = []
        for i, set1 in selected_diseases['name_set'].items():
            for j, set2 in unique_diseases['name_set'].items():
                if set1 == set2:
                    matches.append((i, j))
        indexes = [j for _, j in matches]
        indexes.sort()
        unique_diseases = unique_diseases.iloc[indexes]
        diseases_matched = unique_diseases['disease_name'].to_list()
        return diseases_matched

    def get_dis_pro_data(self, df_dis_gen, df_gen_pro, selected_diseases, output_path):
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
        unique_diseases = pd.DataFrame(
            df_dis_pro['disease_name'].unique()
        ).rename({0: 'disease_name'}, axis=1)
        unique_diseases['disease_name_cleaned'] = unique_diseases['disease_name'].apply(
            lambda x: process_text(x)
        )
        diseases_matched = self.get_matched_diseases(selected_diseases, unique_diseases)
        pd.DataFrame(diseases_matched, columns=['Disease']).to_csv(
            f'{output_path}/diseases_of_interest.csv', index=False
        )
        diseases_matched = diseases_matched[:3]  # esto se borra
        df_dis_pro_matched = df_dis_pro[df_dis_pro['disease_name'].isin(diseases_matched)]
        return df_dis_pro_matched, diseases_matched

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

    def main(self, selected_diseases, output_path):
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
        df_dis_pro_matched, diseases_matched = self.get_dis_pro_data(
            df_dis_gen, df_gen_pro, selected_diseases, output_path
        )
        df_dis_pro_encoded = self.encoding_diseases(df_dis_pro_matched)
        df_pro_pro_encoded, df_dis_pro_encoded = self.encoding_proteins(
            df_pro_pro, df_dis_pro_encoded
        )
        return df_pro_pro_encoded, df_gen_pro, df_dis_gen, df_dis_pro_encoded, diseases_matched
