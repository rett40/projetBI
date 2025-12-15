"""
Configuration centralisée du projet.
Permet d'éviter les chemins absolus dans le code.
"""

from pathlib import Path

# Racine du projet = dossier contenant ce fichier + deux niveaux (scripts/app -> projet)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dossiers importants
DATA_DIR = PROJECT_ROOT / "scripts" / "app" / "data"
OUTPUT_DIR = PROJECT_ROOT

# Fichiers
URLS_CSV = DATA_DIR / "groupe19.csv"
SCRAPED_DATASET = OUTPUT_DIR / "dataset_scraping_nlp.csv"

# Modèle spaCy (multilingue pour mieux détecter les lieux de plusieurs langues)
SPACY_MODEL = "xx_ent_wiki_sm"

# Seuil minimal de mots pour conserver un article (plus bas pour garder plus d'articles)
MIN_WORDS = 30


