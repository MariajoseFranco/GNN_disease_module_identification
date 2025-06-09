import pandas as pd
import requests

# Replace with your actual API key
API_KEY = "API KEY"
BASE_URL = 'http://data.bioontology.org'


def get_cui_from_disease_name(disease_name):
    url = f"{BASE_URL}/search"
    params = {
        'q': disease_name,
        'ontologies': 'MESH',
        'apikey': API_KEY,
        'require_exact_match': 'false'
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    results = response.json().get('collection', [])
    if not results:
        print(f"No results found for '{disease_name}'")
        return None

    for result in results:
        cui = result.get('cui') or result.get('umlscui')
        if cui:
            return cui
        annotations = result.get('annotations', {})
        if 'CUI' in annotations:
            return annotations['CUI']

    print(f"CUI not found in results for '{disease_name}'")
    return None


if __name__ == "__main__":
    diseases = pd.read_csv('./inputs/DIAMOnD_diseases.txt', header=None)[0].unique().tolist()
    df_diseases = pd.DataFrame(columns=['disease_name', 'cui'])
    for disease_name in diseases:
        cui = get_cui_from_disease_name(disease_name)
        df_diseases.loc[len(df_diseases)] = [disease_name, cui[0]]
        df_diseases.to_csv('./inputs/DIAMOnD_diseases_cui.csv', index=False)
