"""
Builds a sample medicine dataset for the prototype.
In production, replace with the full Kaggle A-Z Medicines Dataset of India (253k records).
Dataset: https://www.kaggle.com/datasets/shudhanshusingh/az-medicine-dataset-of-india
"""

import json
import pandas as pd
import os

# ---------------------------------------------------------------------------
# 1. Medicine master — brand → salt(s) mapping
# ---------------------------------------------------------------------------
MEDICINES = [
    # Painkillers / Antipyretics
    {"brand": "Crocin", "salt": "Paracetamol 500mg", "manufacturer": "GSK", "type": "Tablet", "price": 30},
    {"brand": "Calpol", "salt": "Paracetamol 500mg", "manufacturer": "GSK", "type": "Tablet", "price": 28},
    {"brand": "Dolo 650", "salt": "Paracetamol 650mg", "manufacturer": "Micro Labs", "type": "Tablet", "price": 32},
    {"brand": "Tylenol", "salt": "Paracetamol 500mg", "manufacturer": "J&J", "type": "Tablet", "price": 45},
    {"brand": "Combiflam", "salt": "Ibuprofen 400mg + Paracetamol 325mg", "manufacturer": "Sanofi", "type": "Tablet", "price": 36},
    {"brand": "Brufen", "salt": "Ibuprofen 400mg", "manufacturer": "Abbott", "type": "Tablet", "price": 22},
    {"brand": "Ibugesic", "salt": "Ibuprofen 400mg", "manufacturer": "Cipla", "type": "Tablet", "price": 18},
    {"brand": "Nurofen", "salt": "Ibuprofen 200mg", "manufacturer": "Reckitt", "type": "Tablet", "price": 55},
    {"brand": "Aspirin", "salt": "Aspirin 75mg", "manufacturer": "Bayer", "type": "Tablet", "price": 10},
    {"brand": "Ecosprin", "salt": "Aspirin 75mg", "manufacturer": "USV", "type": "Tablet", "price": 12},
    {"brand": "Voveran", "salt": "Diclofenac 50mg", "manufacturer": "Novartis", "type": "Tablet", "price": 28},
    {"brand": "Voltaren", "salt": "Diclofenac 50mg", "manufacturer": "GSK", "type": "Tablet", "price": 40},
    {"brand": "Hifenac", "salt": "Aceclofenac 100mg", "manufacturer": "Intas", "type": "Tablet", "price": 55},
    {"brand": "Zerodol", "salt": "Aceclofenac 100mg", "manufacturer": "Ipca", "type": "Tablet", "price": 50},

    # Antibiotics
    {"brand": "Augmentin", "salt": "Amoxicillin 500mg + Clavulanate 125mg", "manufacturer": "GSK", "type": "Tablet", "price": 180},
    {"brand": "Mox", "salt": "Amoxicillin 500mg", "manufacturer": "Ranbaxy", "type": "Capsule", "price": 45},
    {"brand": "Novamox", "salt": "Amoxicillin 500mg", "manufacturer": "Cipla", "type": "Capsule", "price": 42},
    {"brand": "Azithral", "salt": "Azithromycin 500mg", "manufacturer": "Alembic", "type": "Tablet", "price": 95},
    {"brand": "Zithromax", "salt": "Azithromycin 500mg", "manufacturer": "Pfizer", "type": "Tablet", "price": 140},
    {"brand": "Azee", "salt": "Azithromycin 500mg", "manufacturer": "Cipla", "type": "Tablet", "price": 90},
    {"brand": "Ciprobid", "salt": "Ciprofloxacin 500mg", "manufacturer": "Cadila", "type": "Tablet", "price": 65},
    {"brand": "Ciplox", "salt": "Ciprofloxacin 500mg", "manufacturer": "Cipla", "type": "Tablet", "price": 60},
    {"brand": "Cifran", "salt": "Ciprofloxacin 500mg", "manufacturer": "Ranbaxy", "type": "Tablet", "price": 58},
    {"brand": "Taxim-O", "salt": "Cefixime 200mg", "manufacturer": "Alkem", "type": "Tablet", "price": 120},
    {"brand": "Cefix", "salt": "Cefixime 200mg", "manufacturer": "Cipla", "type": "Tablet", "price": 100},
    {"brand": "Zifi", "salt": "Cefixime 200mg", "manufacturer": "FDC", "type": "Tablet", "price": 110},

    # Antacids / GI
    {"brand": "Omez", "salt": "Omeprazole 20mg", "manufacturer": "Dr. Reddy's", "type": "Capsule", "price": 55},
    {"brand": "Prilosec", "salt": "Omeprazole 20mg", "manufacturer": "AstraZeneca", "type": "Capsule", "price": 90},
    {"brand": "Pan", "salt": "Pantoprazole 40mg", "manufacturer": "Alkem", "type": "Tablet", "price": 48},
    {"brand": "Pantop", "salt": "Pantoprazole 40mg", "manufacturer": "Aristo", "type": "Tablet", "price": 42},
    {"brand": "Nexpro", "salt": "Esomeprazole 40mg", "manufacturer": "Torrent", "type": "Tablet", "price": 65},
    {"brand": "Nexium", "salt": "Esomeprazole 40mg", "manufacturer": "AstraZeneca", "type": "Tablet", "price": 120},
    {"brand": "Rantac", "salt": "Ranitidine 150mg", "manufacturer": "JB Chemicals", "type": "Tablet", "price": 18},
    {"brand": "Zinetac", "salt": "Ranitidine 150mg", "manufacturer": "GSK", "type": "Tablet", "price": 22},

    # Antihypertensives
    {"brand": "Telma", "salt": "Telmisartan 40mg", "manufacturer": "Glenmark", "type": "Tablet", "price": 85},
    {"brand": "Telsartan", "salt": "Telmisartan 40mg", "manufacturer": "Dr. Reddy's", "type": "Tablet", "price": 75},
    {"brand": "Olsar", "salt": "Olmesartan 20mg", "manufacturer": "Sun Pharma", "type": "Tablet", "price": 110},
    {"brand": "Olmy", "salt": "Olmesartan 20mg", "manufacturer": "Cipla", "type": "Tablet", "price": 95},
    {"brand": "Amlodac", "salt": "Amlodipine 5mg", "manufacturer": "Cadila", "type": "Tablet", "price": 38},
    {"brand": "Amlovas", "salt": "Amlodipine 5mg", "manufacturer": "Macleods", "type": "Tablet", "price": 32},
    {"brand": "Norvasc", "salt": "Amlodipine 5mg", "manufacturer": "Pfizer", "type": "Tablet", "price": 120},
    {"brand": "Metolar", "salt": "Metoprolol 50mg", "manufacturer": "Cipla", "type": "Tablet", "price": 45},
    {"brand": "Betaloc", "salt": "Metoprolol 50mg", "manufacturer": "AstraZeneca", "type": "Tablet", "price": 65},

    # Antidiabetics
    {"brand": "Glycomet", "salt": "Metformin 500mg", "manufacturer": "USV", "type": "Tablet", "price": 28},
    {"brand": "Glucophage", "salt": "Metformin 500mg", "manufacturer": "Merck", "type": "Tablet", "price": 55},
    {"brand": "Zoryl", "salt": "Glimepiride 1mg", "manufacturer": "Intas", "type": "Tablet", "price": 35},
    {"brand": "Amaryl", "salt": "Glimepiride 1mg", "manufacturer": "Sanofi", "type": "Tablet", "price": 65},
    {"brand": "Januvia", "salt": "Sitagliptin 100mg", "manufacturer": "MSD", "type": "Tablet", "price": 480},
    {"brand": "Istavel", "salt": "Sitagliptin 100mg", "manufacturer": "Sun Pharma", "type": "Tablet", "price": 280},

    # Vitamins / Supplements
    {"brand": "Limcee", "salt": "Vitamin C 500mg", "manufacturer": "Abbott", "type": "Tablet", "price": 12},
    {"brand": "Celin", "salt": "Vitamin C 500mg", "manufacturer": "GSK", "type": "Tablet", "price": 14},
    {"brand": "Shelcal", "salt": "Calcium 500mg + Vitamin D3 250IU", "manufacturer": "Torrent", "type": "Tablet", "price": 85},
    {"brand": "Calcirol", "salt": "Vitamin D3 60000IU", "manufacturer": "Cadila", "type": "Capsule", "price": 28},

    # Antihistamines
    {"brand": "Allegra", "salt": "Fexofenadine 120mg", "manufacturer": "Sanofi", "type": "Tablet", "price": 120},
    {"brand": "Fexova", "salt": "Fexofenadine 120mg", "manufacturer": "Cipla", "type": "Tablet", "price": 85},
    {"brand": "Cetrizine", "salt": "Cetirizine 10mg", "manufacturer": "Various", "type": "Tablet", "price": 8},
    {"brand": "Zyrtec", "salt": "Cetirizine 10mg", "manufacturer": "UCB", "type": "Tablet", "price": 45},
    {"brand": "Atarax", "salt": "Hydroxyzine 25mg", "manufacturer": "UCB", "type": "Tablet", "price": 38},

    # Cough / Cold
    {"brand": "Benadryl", "salt": "Diphenhydramine 25mg", "manufacturer": "J&J", "type": "Syrup", "price": 85},
    {"brand": "Phenergan", "salt": "Promethazine 25mg", "manufacturer": "Sanofi", "type": "Tablet", "price": 22},
    {"brand": "Alex", "salt": "Dextromethorphan + Guaifenesin + Phenylephrine", "manufacturer": "Glenmark", "type": "Syrup", "price": 75},

    # Cholesterol
    {"brand": "Atorva", "salt": "Atorvastatin 10mg", "manufacturer": "Zydus", "type": "Tablet", "price": 38},
    {"brand": "Lipitor", "salt": "Atorvastatin 10mg", "manufacturer": "Pfizer", "type": "Tablet", "price": 145},
    {"brand": "Storvas", "salt": "Atorvastatin 20mg", "manufacturer": "Sun Pharma", "type": "Tablet", "price": 55},
    {"brand": "Rosuvas", "salt": "Rosuvastatin 10mg", "manufacturer": "Sun Pharma", "type": "Tablet", "price": 85},
    {"brand": "Crestor", "salt": "Rosuvastatin 10mg", "manufacturer": "AstraZeneca", "type": "Tablet", "price": 220},

    # Thyroid
    {"brand": "Eltroxin", "salt": "Levothyroxine 50mcg", "manufacturer": "GSK", "type": "Tablet", "price": 28},
    {"brand": "Thyronorm", "salt": "Levothyroxine 50mcg", "manufacturer": "Abbott", "type": "Tablet", "price": 32},
]

# ---------------------------------------------------------------------------
# 2. Disease → salt mapping
# ---------------------------------------------------------------------------
DISEASE_MAP = {
    "fever": ["Paracetamol 500mg", "Paracetamol 650mg", "Ibuprofen 400mg"],
    "headache": ["Paracetamol 500mg", "Ibuprofen 400mg", "Aspirin 75mg", "Diclofenac 50mg"],
    "pain": ["Ibuprofen 400mg", "Diclofenac 50mg", "Aceclofenac 100mg", "Paracetamol 500mg"],
    "cold": ["Cetirizine 10mg", "Fexofenadine 120mg", "Dextromethorphan + Guaifenesin + Phenylephrine"],
    "cough": ["Dextromethorphan + Guaifenesin + Phenylephrine", "Promethazine 25mg"],
    "allergy": ["Cetirizine 10mg", "Fexofenadine 120mg", "Hydroxyzine 25mg"],
    "acidity": ["Omeprazole 20mg", "Pantoprazole 40mg", "Esomeprazole 40mg", "Ranitidine 150mg"],
    "gastritis": ["Omeprazole 20mg", "Pantoprazole 40mg", "Ranitidine 150mg"],
    "ulcer": ["Pantoprazole 40mg", "Esomeprazole 40mg", "Omeprazole 20mg"],
    "infection": ["Amoxicillin 500mg", "Azithromycin 500mg", "Ciprofloxacin 500mg", "Cefixime 200mg"],
    "bacterial infection": ["Amoxicillin 500mg", "Ciprofloxacin 500mg", "Cefixime 200mg", "Azithromycin 500mg"],
    "throat infection": ["Amoxicillin 500mg", "Azithromycin 500mg", "Cefixime 200mg"],
    "urinary infection": ["Ciprofloxacin 500mg", "Cefixime 200mg"],
    "hypertension": ["Telmisartan 40mg", "Olmesartan 20mg", "Amlodipine 5mg", "Metoprolol 50mg"],
    "high blood pressure": ["Telmisartan 40mg", "Amlodipine 5mg", "Metoprolol 50mg"],
    "diabetes": ["Metformin 500mg", "Glimepiride 1mg", "Sitagliptin 100mg"],
    "type 2 diabetes": ["Metformin 500mg", "Glimepiride 1mg", "Sitagliptin 100mg"],
    "cholesterol": ["Atorvastatin 10mg", "Atorvastatin 20mg", "Rosuvastatin 10mg"],
    "high cholesterol": ["Atorvastatin 10mg", "Rosuvastatin 10mg"],
    "thyroid": ["Levothyroxine 50mcg"],
    "hypothyroidism": ["Levothyroxine 50mcg"],
    "vitamin deficiency": ["Vitamin C 500mg", "Calcium 500mg + Vitamin D3 250IU", "Vitamin D3 60000IU"],
    "calcium deficiency": ["Calcium 500mg + Vitamin D3 250IU"],
    "joint pain": ["Diclofenac 50mg", "Aceclofenac 100mg", "Ibuprofen 400mg"],
    "arthritis": ["Diclofenac 50mg", "Aceclofenac 100mg", "Ibuprofen 400mg"],
}

# ---------------------------------------------------------------------------
# 3. Salt canonical forms / synonyms
# ---------------------------------------------------------------------------
SALT_SYNONYMS = {
    "Paracetamol": ["Acetaminophen", "PCM", "Para", "APAP"],
    "Ibuprofen": ["Advil", "Motrin", "IBU"],
    "Aspirin": ["Acetylsalicylic acid", "ASA"],
    "Diclofenac": ["Voltarol"],
    "Aceclofenac": ["ACL"],
    "Amoxicillin": ["Amox", "AMOX"],
    "Azithromycin": ["AZT", "Azithro"],
    "Ciprofloxacin": ["Cipro", "CIP"],
    "Cefixime": ["Cef"],
    "Omeprazole": ["PPI", "OMP"],
    "Pantoprazole": ["PAN", "PPZ"],
    "Esomeprazole": ["ESO"],
    "Ranitidine": ["RAN", "H2 blocker"],
    "Metformin": ["MET", "Glucophage generic"],
    "Atorvastatin": ["ATV", "Statin"],
    "Rosuvastatin": ["RSV", "Statin"],
    "Telmisartan": ["TEL", "ARB"],
    "Amlodipine": ["AML", "CCB"],
    "Metoprolol": ["MET", "Beta blocker"],
    "Levothyroxine": ["LT4", "T4", "L-thyroxine"],
    "Cetirizine": ["CTZ", "Antihistamine"],
    "Fexofenadine": ["FEX"],
}

def build_and_save():
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(MEDICINES)
    df.to_csv("data/medicines.csv", index=False)
    with open("data/disease_map.json", "w") as f:
        json.dump(DISEASE_MAP, f, indent=2)
    with open("data/salt_synonyms.json", "w") as f:
        json.dump(SALT_SYNONYMS, f, indent=2)
    print(f"✓ Dataset built: {len(df)} medicines, {len(DISEASE_MAP)} diseases, {len(SALT_SYNONYMS)} synonym groups")
    return df

if __name__ == "__main__":
    build_and_save()
