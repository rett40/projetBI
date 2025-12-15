import re
from collections import Counter
from typing import List, Dict, Any

import pandas as pd
import spacy
from langdetect import detect, LangDetectException
from textblob import TextBlob

from .config import SPACY_MODEL


_NLP = None


def get_nlp():
    """
    Chargement paresseux du modèle spaCy.
    On évite de le charger plusieurs fois.
    """
    global _NLP
    if _NLP is None:
        _NLP = spacy.load(SPACY_MODEL)
    return _NLP


# Dictionnaires métiers (anglais principalement, avec normalisation multilingue)
DISEASE_LIST = [
    "flu", "covid-19", "coronavirus", "dengue", "anthrax", "measles",
    "cholera", "ebola", "malaria", "tuberculosis", "cancer", "fever",
    "avian flu", "foot and mouth", "rift valley fever", "rabies", "brucellosis",
]

ANIMALS = [
    "cow", "chicken", "goat", "sheep", "camel", "bird",
    "poultry", "pig", "dog", "cat", "horse",
]

# Variantes multi-langues → forme canonique (clé)
DISEASE_VARIANTS = {
    "flu": ["flu", "influenza", "grippe", "انفلونزا", "الانفلونزا"],
    "covid-19": [
        "covid", "covid-19", "coronavirus", "sars-cov-2",
        "كورونا", "فيروس كورونا", "كوفيد", "كوفيد-19",
    ],
    "malaria": ["malaria", "paludisme", "ملاريا", "الملاريا"],
    "cholera": ["cholera", "كوليرا", "الكوليرا"],
    "fever": ["fever", "fièvre", "حمى"],
    "rift valley fever": ["rift valley fever", "حمى الوادي المتصدع"],
    "rabies": ["rabies", "rage", "داء الكلب", "السعار"],
    "cancer": ["cancer", "سرطان"],
}

ANIMAL_VARIANTS = {
    "cat": ["cat", "chat", "قط", "قطط"],
    "dog": ["dog", "chien", "كلب", "كلاب"],
    "cow": ["cow", "cattle", "vache", "أبقار", "ابقار", "بقر", "بقرة"],
    "chicken": ["chicken", "poulet", "دجاج"],
    "poultry": ["poultry", "volaille", "دواجن"],
    "goat": ["goat", "chèvre", "ماعز"],
    "sheep": ["sheep", "mouton", "خراف", "أغنام", "اغنام"],
    "camel": ["camel", "chameau", "جمال", "إبل", "ابل"],
    "pig": ["pig", "swine", "porc", "خنازير", "خنزير"],
    "horse": ["horse", "cheval", "خيول", "حصان"],
    "bird": ["bird", "oiseau", "طيور"],
}

AR_COUNTRY_TERMS = {
    "تونس": "Tunisia",
    "المغرب": "Morocco",
    "الجزائر": "Algeria",
    "مصر": "Egypt",
    "ليبيا": "Libya",
    "العراق": "Iraq",
    "سوريا": "Syria",
    "لبنان": "Lebanon",
    "الأردن": "Jordan",
    "السعودية": "Saudi Arabia",
    "اليمن": "Yemen",
    "الكويت": "Kuwait",
    "قطر": "Qatar",
    "الإمارات": "United Arab Emirates",
    "الامارات": "United Arab Emirates",
}


def clean_text(text: str) -> str:
    """Nettoyage léger du texte."""
    if not isinstance(text, str):
        return ""
    # Supprimer espaces multiples, retours à la ligne
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except (LangDetectException, TypeError):
        return "unknown"


def has_arabic(text: str) -> bool:
    """Détecte si le texte contient des caractères arabes."""
    if not isinstance(text, str):
        return False
    return bool(re.search(r"[\u0600-\u06FF]", text))


def detect_diseases_nlp(text: str) -> List[str]:
    """
    Utilise un dictionnaire + lemmatisation spaCy + recherche d'expressions.
    """
    text_low = text.lower()
    found = set()

    # 1) Normalisation par dictionnaire multi-langues
    for canon, variants in DISEASE_VARIANTS.items():
        for v in variants:
            if v in text_low or v in text:
                found.add(canon)

    # 2) Fallback spaCy (surtout utile pour l'anglais ou langues proches)
    if not found:
        nlp = get_nlp()
        doc = nlp(text_low)
        for token in doc:
            lemma = token.lemma_.lower()
            if lemma in DISEASE_LIST:
                found.add(lemma)

    return sorted(found)


def detect_locations(text: str) -> List[str]:
    """
    Reconnaît les lieux via spaCy NER.
    """
    nlp = get_nlp()
    doc = nlp(text)
    locations = {ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]}

    # Complément pour l'arabe : recherche directe de noms de pays arabes
    if has_arabic(text):
        for ar_name, en_name in AR_COUNTRY_TERMS.items():
            if ar_name in text:
                locations.add(en_name)

    return sorted(locations)


def detect_organisations(text: str) -> List[str]:
    nlp = get_nlp()
    doc = nlp(text)
    return sorted({ent.text for ent in doc.ents if ent.label_ == "ORG"})


def detect_animals(text: str) -> List[str]:
    """
    Détection simple via dictionnaire, insensible à la casse.
    """
    text_low = text.lower()
    found = set()

    # Normalisation multi-langues -> forme canonique
    for canon, variants in ANIMAL_VARIANTS.items():
        for v in variants:
            if v in text_low or v in text:
                found.add(canon)

    return sorted(found)


def detect_dates(text: str) -> List[str]:
    """
    Détection de dates dans le contenu, normalisées au format jj-mm-aaaa.
    """
    nlp = get_nlp()
    doc = nlp(text)
    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]

    parsed_dates = []
    for d in dates:
        try:
            dt = pd.to_datetime(d, errors="coerce", dayfirst=True)
            if not pd.isna(dt):
                parsed_dates.append(dt.strftime("%d-%m-%Y"))
        except Exception:
            continue

    return sorted(set(parsed_dates))


def detect_publication_source(text: str) -> str:
    """
    Détection heuristique de la source à partir de patterns dans le texte.
    """
    text_low = text.lower()

    if "facebook" in text_low or "posted on social media" in text_low:
        return "Social Media"
    if "twitter" in text_low or "x.com" in text_low:
        return "Social Media"
    if "ministry" in text_low or "ministère" in text_low or "gov" in text_low:
        return "Official Source"
    if "press" in text_low or "journal" in text_low or "news agency" in text_low:
        return "Media"
    if "tv" in text_low or "radio" in text_low or "television" in text_low:
        return "Media (TV/Radio)"

    return "Unknown"


def summarize(text: str, word_limit: int = 50) -> str:
    words = text.split()
    if len(words) <= word_limit:
        return text
    return " ".join(words[:word_limit])


def compute_sentiment(text: str) -> float:
    """
    Score de sentiment simple via TextBlob (-1 à 1).
    """
    if not text:
        return 0.0
    try:
        return float(TextBlob(text).sentiment.polarity)
    except Exception:
        return 0.0


def build_record(
    url: str,
    title: str,
    text: str,
    publication_date,
) -> Dict[str, Any]:
    """
    Construit un enregistrement enrichi à partir du texte brut.
    """
    text = clean_text(text)
    lang = detect_language(text)
    word_count = len(text.split())
    char_count = len(text)

    diseases = detect_diseases_nlp(text)
    locations = detect_locations(text)
    orgs = detect_organisations(text)
    animals = detect_animals(text)
    dates_in_text = detect_dates(text)
    source_nlp = detect_publication_source(text)
    sentiment = compute_sentiment(text)

    sum_50 = summarize(text, 50)
    sum_100 = summarize(text, 100)
    sum_150 = summarize(text, 150)

    return {
        "url": url,
        "title": title,
        "text": text,
        "language": lang,
        "char_count": char_count,
        "word_count": word_count,
        "publication_date_detected": publication_date.strftime("%d-%m-%Y")
        if publication_date
        else None,
        "dates_mentioned": ";".join(dates_in_text),
        "locations": ";".join(locations),
        "organisations": ";".join(orgs),
        "animals": ";".join(animals),
        "diseases": ";".join(diseases),
        "source_nlp": source_nlp,
        "sentiment": sentiment,
        "summary_50": sum_50,
        "summary_100": sum_100,
        "summary_150": sum_150,
    }


