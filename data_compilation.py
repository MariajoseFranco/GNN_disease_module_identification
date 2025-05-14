import pandas as pd


class DataCompilation():
    def __init__(self, path) -> None:
        self.path = path

    def get_diseases(self):
        diseases = pd.read_csv('./inputs/diseases.txt', header=None)
        diseases = diseases.rename({0: 'DISEASE'}, axis=1)
        diseases['DISEASE'] = diseases['DISEASE'].apply(lambda x: x.strip().lower().split())
        return diseases

    def get_matched_diseases(self, selected_diseases, unique_diseases):
        # Convert to sets for comparison
        selected_diseases['name_set'] = selected_diseases['DISEASE'].apply(set)
        unique_diseases['name_set'] = unique_diseases['disease_name_cleaned'].apply(set)

        # Create a list to store matches
        matches = []

        # Compare each set in df1 to all sets in df2
        for i, set1 in selected_diseases['name_set'].items():
            for j, set2 in unique_diseases['name_set'].items():
                if set1 == set2:
                    matches.append((i, j))
        indexes = [j for _, j in matches]
        indexes.sort()
        unique_diseases = unique_diseases.iloc[indexes]
        diseases_matched = unique_diseases['disease_name'].to_list()
        return diseases_matched

    def get_data(self):
        # Protein - Protein Interaction
        df_pro_pro = pd.read_csv(f'{self.path}pro_pro.tsv', sep='\t')
        df_pro_pro = df_pro_pro[df_pro_pro['prA'] != df_pro_pro['prB']]
        # Gen - Protein Interaction
        df_gen_pro = pd.read_csv(f'{self.path}gen_pro.tsv', sep='\t')
        # Disease - Gen Interaction
        df_dis_gen = pd.read_csv(f'{self.path}dis_gen.tsv', sep='\t')
        return df_pro_pro, df_gen_pro, df_dis_gen

    def get_dis_pro_data(self, df_dis_gen, df_gen_pro, selected_diseases):
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
            lambda x: x.strip().lower().split()
        )
        diseases_matched = self.get_matched_diseases(selected_diseases, unique_diseases)
        df_dis_pro = df_dis_pro[df_dis_pro['disease_name'].isin(diseases_matched)]
        df_dis_pro_matched = df_dis_pro[df_dis_pro['disease_name'].isin(diseases_matched)]
        return df_dis_pro_matched, diseases_matched

    def main(self, selected_diseases):
        df_pro_pro, df_gen_pro, df_dis_gen = self.get_data()
        df_dis_pro, diseases_matched = self.get_dis_pro_data(
            df_dis_gen, df_gen_pro, selected_diseases
        )
        return df_pro_pro, df_gen_pro, df_dis_gen, df_dis_pro, diseases_matched
