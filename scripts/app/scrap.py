import logging
from typing import Tuple, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from newspaper import Article

from .config import URLS_CSV, SCRAPED_DATASET, MIN_WORDS
from .nlp_utils import build_record, clean_text


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================
# 1. Scraping
# ============================

def scrape_text(url: str) -> Tuple[Optional[str], Optional[str], Optional[pd.Timestamp]]:
    """
    Extraction de texte à partir d'une URL.
    Priorité à Newspaper3k puis fallback BeautifulSoup.
    """
    # 1) Newspaper3k
    try:
        article = Article(url)
        article.download()
        article.parse()

        if article.text and len(article.text.split()) >= 10:
            return article.title, article.text, article.publish_date
    except Exception as e:
        logger.debug(f"Newspaper3k a échoué pour {url}: {e}")

    # 2) Fallback BeautifulSoup
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n".join([p.get_text(strip=True) for p in paragraphs])
        title = soup.title.text.strip() if soup.title else None
        text = clean_text(text)

        # On laisse la décision finale au filtre MIN_WORDS
        if len(text.split()) < 10:
            return title, None, None

        return title, text, None
    except Exception as e:
        logger.warning(f"Erreur BeautifulSoup pour {url}: {e}")
        return None, None, None


# ============================
# 2. Pipeline principal
# ============================

def run_pipeline():
    logger.info(f"Chargement des URLs depuis {URLS_CSV}")
    df_urls = pd.read_csv(URLS_CSV)

    results = []
    total_urls = len(df_urls)
    failed_count = 0
    too_short_count = 0

    for _, row in df_urls.iterrows():
        url = row.get("lien") or row.get("url")
        if not url:
            continue

        logger.info(f"Scraping : {url}")

        title, text, pub_date = scrape_text(url)

        if not text:
            logger.warning(f"Erreur extraction pour {url} (texte vide)")
            failed_count += 1
            results.append(
                {
                    "url": url,
                    "title": title,
                    "text": None,
                    "language": None,
                    "char_count": None,
                    "word_count": None,
                    "publication_date_detected": None,
                    "dates_mentioned": "",
                    "locations": "",
                    "organisations": "",
                    "animals": "",
                    "diseases": "",
                    "source_nlp": None,
                    "sentiment": None,
                    "summary_50": None,
                    "summary_100": None,
                    "summary_150": None,
                    "scrape_status": "scrape_failed",
                }
            )
            continue

        # Filtrer les articles trop courts
        if len(text.split()) < MIN_WORDS:
            logger.info(f"Article ignoré (trop court) pour {url}")
            too_short_count += 1
            results.append(
                {
                    "url": url,
                    "title": title,
                    "text": text,
                    "language": None,
                    "char_count": len(text),
                    "word_count": len(text.split()),
                    "publication_date_detected": None,
                    "dates_mentioned": "",
                    "locations": "",
                    "organisations": "",
                    "animals": "",
                    "diseases": "",
                    "source_nlp": None,
                    "sentiment": None,
                    "summary_50": None,
                    "summary_100": None,
                    "summary_150": None,
                    "scrape_status": "too_short",
                }
            )
            continue

        record = build_record(url, title, text, pub_date)
        record["scrape_status"] = "ok"
        results.append(record)

    if not results:
        logger.error("Aucun article valide n'a été extrait.")
        return

    df_final = pd.DataFrame(results)

    # Suppression des doublons éventuels
    df_final = df_final.drop_duplicates(subset=["url"], keep="first")

    df_final.to_csv(SCRAPED_DATASET, index=False)
    logger.info(f"✔ Dataset généré : {SCRAPED_DATASET}")
    logger.info(
        f"Résumé scraping - total URLs: {total_urls}, succès: {total_urls - failed_count}, "
        f"échecs scraping: {failed_count}, articles trop courts: {too_short_count}"
    )


if __name__ == "__main__":
    run_pipeline()
