from typing import Union

import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DataCompilation():
    def __init__(self, data_path, disease_path, output_path) -> None:
        self.data_path = data_path
        self.disease_path = disease_path
        self.output_path = output_path

    def get_data(self) -> Union[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Retrieve and clean the data from the specified paths.

        Returns:
            tuple:
                - pd.DataFrame: Protein-Protein interaction data.
                - pd.DataFrame: Gene-Protein interaction data.
                - pd.DataFrame: Disease-Gene interaction data.
        """
        df_pro_pro = self.get_and_clean_pro_pro_data()
        df_gen_pro = self.get_and_clean_gen_pro_data()
        df_dis_gen = self.get_and_clean_dis_gen_data()
        return df_pro_pro, df_gen_pro, df_dis_gen

    def get_and_clean_pro_pro_data(self) -> pd.DataFrame:
        """_summary_

        Returns:
            pd.DataFrame: _description_
        """
        df_ppi = pd.read_csv(f'{self.data_path}/pro_pro.tsv', sep='\t')

        df_ppi = df_ppi[df_ppi['prA'] != df_ppi['prB']]
        df_ppi = df_ppi.drop_duplicates()
        df_ppi['pair'] = df_ppi.apply(lambda row: tuple(sorted([row['prA'], row['prB']])), axis=1)
        df_pro_pro = (
            df_ppi
            .drop_duplicates(subset='pair')
            .drop(columns='pair').reset_index(drop=True)
        )
        return df_pro_pro

    def get_and_clean_gen_pro_data(self) -> pd.DataFrame:
        """_summary_

        Returns:
            pd.DataFrame: _description_
        """
        df_gen_pro = pd.read_csv(f'{self.data_path}/gen_pro.tsv', sep='\t')
        df_gen_pro = df_gen_pro.drop_duplicates()
        return df_gen_pro

    def get_and_clean_dis_gen_data(self) -> pd.DataFrame:
        """_summary_

        Returns:
            pd.DataFrame: _description_
        """
        df_dis_gen = pd.read_csv(f'{self.data_path}/dis_gen.tsv', sep='\t')
        df_dis_gen = df_dis_gen.drop_duplicates(subset=['disease_name', 'gene_id', 'gene_symbol'])
        return df_dis_gen

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
        )  # se eliminan los genes que no codifican proteina (no salen en df_gen_pro)
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

    def get_matched_diseases(self, df_dis_pro):
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
        with open(
            f"{self.output_path}/diseases_of_interest.csv", "w"
        ) as f:
            f.write("DISEASES OF INTEREST\n")
            for disease in df_dis_pro_matched['disease_name'].unique():
                f.write(f"{disease}\n")
        selected_diseases = df_dis_pro_matched['disease_name'].unique().tolist()
        return df_dis_pro_matched, selected_diseases

    def removing_duplicated_proteins(self, df_dis_pro, selected_diseases):
        """
        Remove duplicated

        Args:
            df_dis_pro (pd.DataFrame): DataFrame containing the disease-protein
            associations.

        Returns:
            pd.DataFrame: DataFrame with duplicated interactions removed.
        """
        results = []
        for disease in selected_diseases:
            df_disease = df_dis_pro[df_dis_pro['disease_name'] == disease]
            df_disease_without_duplicated_proteins = df_disease \
                .sort_values('score', ascending=False) \
                .drop_duplicates(subset='protein_id', keep='first') \
                .reset_index(drop=True)
            results.append(df_disease_without_duplicated_proteins)
        df_dis_pro_without_duplicated_proteins = pd.concat(results, ignore_index=True)
        return df_dis_pro_without_duplicated_proteins

    def main(self) -> Union[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        df_dis_pro = self.get_dis_pro_data(df_dis_gen, df_gen_pro)
        df_dis_pro_encoded = self.encoding_diseases(df_dis_pro)
        df_pro_pro_encoded, df_dis_pro_encoded = self.encoding_proteins(
            df_pro_pro, df_dis_pro_encoded
        )
        df_dis_pro_matched, selected_diseases = self.get_matched_diseases(
            df_dis_pro_encoded
        )
        df_dis_pro_cleaned = self.removing_duplicated_proteins(
            df_dis_pro_matched, selected_diseases
        )
        return df_pro_pro_encoded, df_gen_pro, df_dis_gen, df_dis_pro_cleaned, selected_diseases
