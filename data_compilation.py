import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DataCompilation():
    """
        Initialize the DataCompilation class with paths to input and output data.

        Args:
            data_path (str): Path to the directory containing interaction datasets.
            disease_path (str): Path to the CSV file with selected diseases (by CUI).
            output_path (str): Path where processed data will be saved.
        """
    def __init__(self, data_path, disease_path, output_path) -> None:
        self.data_path = data_path
        self.disease_path = disease_path
        self.output_path = output_path

    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Executes the data getter pipeline for all three datasets:
        protein-protein, gene-protein, and disease-gene interactions.

        Returns:
            tuple:
                - pd.DataFrame: Cleaned protein-protein interaction data.
                - pd.DataFrame: Cleaned gene-protein interaction data.
                - pd.DataFrame: Cleaned disease-gene interaction data.
        """
        df_pro_pro = self.get_and_clean_pro_pro_data()
        df_gen_pro = self.get_and_clean_gen_pro_data()
        df_dis_gen = self.get_and_clean_dis_gen_data()
        return df_pro_pro, df_gen_pro, df_dis_gen

    def get_and_clean_pro_pro_data(self) -> pd.DataFrame:
        """
        Load and clean protein-protein interaction data by removing duplicates
        and self-interactions.

        Returns:
            pd.DataFrame: Cleaned protein-protein interaction data.
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
        """
        Load and clean gene-protein interaction data by removing duplicates.

        Returns:
            pd.DataFrame: Cleaned gene-protein interaction data.
        """
        df_gen_pro = pd.read_csv(f'{self.data_path}/gen_pro.tsv', sep='\t')
        df_gen_pro = df_gen_pro.drop_duplicates()
        return df_gen_pro

    def get_and_clean_dis_gen_data(self) -> pd.DataFrame:
        """
        Load and clean disease-gene interaction data by removing duplicate disease-gene mappings.

        Returns:
            pd.DataFrame: Cleaned disease-gene interaction data.
        """
        df_dis_gen = pd.read_csv(f'{self.data_path}/dis_gen.tsv', sep='\t')
        df_dis_gen = df_dis_gen.drop_duplicates(subset=['disease_name', 'gene_id', 'gene_symbol'])
        return df_dis_gen

    def get_dis_pro_data(self, df_dis_gen: pd.DataFrame, df_gen_pro: pd.DataFrame) -> pd.DataFrame:
        """
        Merge disease-gene and gene-protein data to obtain disease-protein interactions.
        Filters out genes that do not code for proteins.

        Args:
            df_dis_gen (pd.DataFrame): Disease-Gene interaction data.
            df_gen_pro (pd.DataFrame): Gene-Protein interaction data.

        Returns:
            pd.DataFrame: Merged and filtered disease-protein interaction data.
        """
        df_dis_pro = df_dis_gen.merge(
            df_gen_pro, how='left', on='gene_id', indicator=True
        )
        df_dis_pro = df_dis_pro[df_dis_pro['_merge'] == 'both'].drop(
            ['_merge'], axis=1
        )
        return df_dis_pro

    def encoding_diseases(self, df_dis_pro: pd.DataFrame) -> pd.DataFrame:
        """
        Encode disease names into integer IDs using sklearn's LabelEncoder.

        Args:
            df_dis_pro (pd.DataFrame): Disease-protein interaction data.

        Returns:
            pd.DataFrame: DataFrame with an added 'disease_id' column with the diseases encoded.
        """
        disease_encoder = LabelEncoder()
        df_dis_pro['disease_id'] = disease_encoder.fit_transform(df_dis_pro['disease_name'])
        return df_dis_pro

    def encoding_proteins(
            self, df_pro_pro: pd.DataFrame, df_dis_pro: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Encode protein identifiers into integer IDs using sklearn's LabelEncoder.

        Args:
            df_pro_pro (pd.DataFrame): Protein-protein interaction data.
            df_dis_pro (pd.DataFrame): Disease-protein interaction data.

        Returns:
            tuple:
                - pd.DataFrame: Protein-protein interaction data with 'src_id' and 'dst_id' columns.
                - pd.DataFrame: Disease-protein interaction data with 'protein_id_enc' column.
        """
        protein_encoder = LabelEncoder()
        all_proteins = pd.concat([df_pro_pro['prA'], df_pro_pro['prB'], df_dis_pro['protein_id']])
        protein_encoder.fit(all_proteins)
        df_pro_pro['src_id'] = protein_encoder.transform(df_pro_pro['prA'])
        df_pro_pro['dst_id'] = protein_encoder.transform(df_pro_pro['prB'])
        df_dis_pro['protein_id_enc'] = protein_encoder.transform(df_dis_pro['protein_id'])
        return df_pro_pro, df_dis_pro

    def get_matched_diseases(self, df_dis_pro: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """
        Filter disease-protein interactions to keep only selected diseases (based on CUI list).
        Also writes the matched disease names to a CSV file.

        Args:
            df_dis_pro (pd.DataFrame): Encoded disease-protein interaction data.
            Must include a 'cui' and 'disease_name' column.

        Returns:
            tuple:
                - pd.DataFrame: Filtered disease-protein interaction data for matched diseases.
                - list[str]: List of matched disease names.
        """
        diseases = pd.read_csv(self.disease_path)
        cui_list = diseases['cui'].to_list()
        df_dis_pro_matched = df_dis_pro[df_dis_pro['cui'].isin(cui_list)]
        selected_diseases = df_dis_pro_matched['disease_name'].unique().tolist()
        self.save_matched_diseases_to_csv(df_dis_pro_matched)
        return df_dis_pro_matched, selected_diseases

    def save_matched_diseases_to_csv(self, df_dis_pro_matched: pd.DataFrame) -> None:
        """
        Save the list of unique matched disease names to a CSV file.

        The file will be created in the output path specified during initialization,
        and will include a header followed by the disease names.

        Args:
            df_dis_pro_matched (pd.DataFrame): DataFrame containing disease-protein interactions
            for the matched diseases.

        Returns:
            None
        """
        with open(
            f"{self.output_path}/diseases_of_interest.csv", "w"
        ) as f:
            f.write("DISEASES OF INTEREST\n")
            for disease in df_dis_pro_matched['disease_name'].unique():
                f.write(f"{disease}\n")

    def removing_duplicated_proteins(
            self, df_dis_pro: pd.DataFrame, selected_diseases: list[str]
    ) -> pd.DataFrame:
        """
        Remove duplicate protein associations for each disease by keeping the highest scoring one.

        Args:
            df_dis_pro (pd.DataFrame): Disease-protein interaction data.
            selected_diseases (list[str]): List of disease names to process.

        Returns:
            pd.DataFrame: Disease-protein data with duplicates removed per disease.
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

    def main(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
        """
        Run the full data compilation and processing pipeline:
            - Load raw interaction datasets.
            - Merge disease-gene and gene-protein interactions into disease-protein.
            - Encode diseases and proteins.
            - Filter interactions to retain only selected diseases.
            - Remove duplicated protein associations.

        Returns:
            tuple:
                - pd.DataFrame: Encoded protein-protein interaction data.
                - pd.DataFrame: Cleaned gene-protein interaction data.
                - pd.DataFrame: Cleaned disease-gene interaction data.
                - pd.DataFrame: Final encoded disease-protein interaction data.
                - list[str]: List of matched disease names.
        """
        df_pro_pro, df_gen_pro, df_dis_gen = self.get_data()
        df_dis_pro = self.get_dis_pro_data(df_dis_gen, df_gen_pro)
        df_dis_pro_matched, selected_diseases = self.get_matched_diseases(
            df_dis_pro
        )
        df_dis_pro_cleaned = self.removing_duplicated_proteins(
            df_dis_pro_matched, selected_diseases
        )
        df_dis_pro_encoded = self.encoding_diseases(df_dis_pro_cleaned)
        df_pro_pro_encoded, df_dis_pro_encoded = self.encoding_proteins(
            df_pro_pro, df_dis_pro_encoded
        )
        return df_pro_pro_encoded, df_gen_pro, df_dis_gen, df_dis_pro_encoded, selected_diseases
