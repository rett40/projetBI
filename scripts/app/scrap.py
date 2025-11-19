import pandas as pd
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from langdetect import detect
import re
from collections import Counter
import spacy

# ============================
# 1. Charger le fichier URLs
# ============================
df_urls = pd.read_csv("C:\\Users\\Aziz\\PycharmProjects\\projetBI\\scripts\\app\\data\\groupe19.csv")
print(df_urls.columns)


# Charger modèle spaCy
nlp = spacy.load("en_core_web_sm")

# LISTE de maladies (tu peux enrichir)
disease_list = [
    "flu", "covid", "covid-19", "dengue", "anthrax", "measles",
    "cholera", "ebola", "malaria", "tuberculosis", "cancer",
    "fever", "avian flu", "foot and mouth", "rift valley"
]

# Extraction du pays depuis l’URL
def extract_country(url):
    patterns = {
        "eg": "Egypt",
        "ma": "Morocco",
        "dz": "Algeria",
        "tn": "Tunisia",
        "ly": "Libya",
        "fr": "France",
        "uk": "United Kingdom",
        "us": "United States",
        "ke": "Kenya",
    }
    for key, country in patterns.items():
        if f".{key}" in url:
            return country
    return "Unknown"


# Extraction des maladies dans un texte
def detect_diseases(text):
    text_lower = text.lower()
    found = [d for d in disease_list if d in text_lower]
    return list(set(found))


# Extraire les mots-clés (simple)
def extract_keywords(text, n=10):
    words = re.findall(r"[A-Za-zÀ-ÿ]+", text.lower())
    freq = Counter(words)
    return [w for w, c in freq.most_common(n)]


# =============================
# 2. Scraping + NLP automatisé
# =============================
results = []

for idx, row in df_urls.iterrows():
    code = row["code"]
    url = row["lien"]

    print(f"Scraping → {code} | {url}")

    try:
        # Extraction intelligente via newspaper3k
        article = Article(url)
        article.download()
        article.parse()

        text = article.text
        title = article.title

        # NLP
        lang = detect(text)
        diseases = detect_diseases(text)
        keywords = extract_keywords(text, 10)
        media = re.findall(r"https?://([^/]+)/", url)[0]
        country = extract_country(url)
        word_count = len(text.split())

        # Sauvegarde ligne
        results.append({
            "code": code,
            "url": url,
            "media": media,
            "country": country,
            "language": lang,
            "title": title,
            "text": text,
            "word_count": word_count,
            "keywords": ";".join(keywords),
            "diseases": ";".join(diseases)
        })

    except Exception as e:
        print(f"❌ Erreur pour {url}: {e}")
        results.append({
            "code": code,
            "url": url,
            "media": None,
            "country": None,
            "language": None,
            "title": None,
            "text": None,
            "word_count": 0,
            "keywords": None,
            "diseases": None
        })


# ============================
# 3. Créer le dataset final
# ============================
df_final = pd.DataFrame(results)
df_final.to_csv("dataset_scraping_nlp.csv", index=False)

print("\n🎉 Dataset final généré : dataset_scraping_nlp.csv")
df_final.head()
