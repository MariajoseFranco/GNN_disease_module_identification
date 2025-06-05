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
    diseases = [
        "adrenal gland diseases",
        "alzheimer disease",
        "Amino acid metabolism inborn errors",
        "amyotrophic lateral sclerosis",
        "anemia aplastic",
        "anemia hemolytic",
        "aneurysm",
        "arrhythmias cardiac",
        "arthritis rheumatoid",
        "asthma",
        "arterial occlusive diseases",
        "arteriosclerosis",
        "basal ganglia diseases",
        "behcet syndrome",
        "bile duct diseases",
        "blood coagulation disorders",
        "blood platelet disorders",
        "breast neoplasms",
        "carbohydrate metabolism inborn errors",
        "carcinoma renal cell",
        "cardiomyopathies",
        "cardiomyopathy hypertrophic",
        "celiac disease",
        "cerebellar ataxia",
        "cerebrovascular disorders",
        "charcot-marie-tooth disease",
        "colitis ulcerative",
        "colorectal neoplasms",
        "coronary artery disease",
        "crohn disease",
        "death sudden",
        "diabetes mellitus type 2",
        "dwarfism",
        "esophageal diseases",
        "exophthalmos",
        "glomerulonephritis",
        "gout",
        "graves disease",
        "head and neck neoplasms",
        "hypothalamic diseases",
        "leukemia b-cell",
        "leukemia myeloid",
        "lipid metabolism disorders",
        "liver cirrhosis",
        "liver cirrhosis biliary",
        "Lung diseases obstructive",
        "lupus erythematosus",
        "lymphoma",
        "lysosomal storage diseases",
        "macular degeneration",
        "metabolic syndrome x",
        "motor neuron disease",
        "multiple sclerosis",
        "muscular dystrophies",
        "mycobacterium infections",
        "myeloproliferative disorders",
        "metabolic and nutritional diseases",
        "peroxisomal disorders",
        "psoriasis",
        "purine-pyrimidine metabolism inborn errors",
        "renal tubular transport inborn errors",
        "sarcoma",
        "spastic paraplegia hereditary",
        "spinocerebellar ataxias",
        "spinocerebellar degenerations",
        "spondylarthropathies",
        "tauopathies",
        "uveal diseases",
        "varicose veins",
        "vasculitis",
    ]
    df_diseases = pd.DataFrame(columns=['disease_name', 'cui'])
    for disease_name in diseases:
        cui = get_cui_from_disease_name(disease_name)
        df_diseases.loc[len(df_diseases)] = [disease_name, cui[0]]
        df_diseases.to_csv('./inputs/diseases_cui.csv', index=False)
