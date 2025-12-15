import streamlit as st
import pandas as pd
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Dashboard - Données Santé Animale",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
.main-header { 
    font-size: 2.5rem; 
    color: #1f77b4; 
    text-align: center; 
    margin-bottom: 2rem;
    font-weight: bold;
}
.section-header { 
    font-size: 1.4rem; 
    color: #1f77b4; 
    margin-top: 2rem; 
    margin-bottom: 1rem;
    font-weight: 600;
}
.kpi-card { 
    background: #f0f2f6; 
    padding: 1.5rem; 
    border-radius: 10px; 
    border-left: 4px solid #1f77b4;
    text-align: center;
}
.kpi-value {
    font-size: 1.8rem;
    font-weight: bold;
    color: #1f77b4;
}
.kpi-label {
    font-size: 0.9rem;
    color: #666;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 Dashboard Analytique - Données Santé Animale</div>', unsafe_allow_html=True)

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dataset_scraping_nlp.csv")

        # Nettoyage colonnes
        df["language"] = df.get("language", "").fillna("Unknown")
        df["source_nlp"] = df.get("source_nlp", "").fillna("Unknown")
        df["locations"] = df.get("locations", "").fillna("Unknown")
        df["diseases"] = df.get("diseases", "").fillna("")
        df["animals"] = df.get("animals", "").fillna("")

        # Transformer "locations" → liste
        df["location_list"] = df["locations"].apply(
            lambda x: [l.strip() for l in str(x).split(";") if l.strip()]
        )

        # Détection améliorée des pays
        def detect_country(loc_list):
            known_countries = [
                "Tunisia", "France", "Morocco", "USA", "United States", "India", "Egypt",
                "UK", "United Kingdom", "Spain", "China", "Italy", "Sweden", "Iraq",
                "Libya", "Pakistan", "Turkey", "Brazil", "Afghanistan", "Vietnam",
                "Canada", "Australia", "South Africa", "Kenya", "Nigeria", "Mexico",
                "Germany", "Netherlands", "Belgium", "Switzerland", "Portugal", "Algeria",
                "Saudi Arabia", "Syria", "Lebanon", "Jordan", "Yemen", "Kuwait",
                "Qatar", "United Arab Emirates",
            ]

            # Variantes et abréviations vers forme canonique
            country_mapping = {
                "US": "USA", "United States": "USA", "U.S.": "USA", "America": "USA",
                "USA": "USA",
                "UK": "United Kingdom", "U.K.": "United Kingdom", "Britain": "United Kingdom",
                "UAE": "United Arab Emirates", "Emirates": "United Arab Emirates",
                "États-Unis": "USA", "Etats-Unis": "USA", "États Unis": "USA",
                "Tunisie": "Tunisia", "Maroc": "Morocco", "Egypte": "Egypt", "Égypte": "Egypt",
            }

            for loc in loc_list[::-1]:
                loc_clean = loc.strip()
                if not loc_clean:
                    continue
                # Vérifier le mapping d'abord
                if loc_clean in country_mapping:
                    return country_mapping[loc_clean]
                # Vérifier les pays connus
                if loc_clean in known_countries:
                    return loc_clean

            # Si aucun match, considérer comme inconnu pour éviter des pays "bizarres"
            return "Unknown"

        df["country"] = df["location_list"].apply(detect_country)

        # Transformer "diseases" en liste
        df["disease_list"] = df["diseases"].apply(
            lambda x: [d.strip() for d in str(x).split(";") if d.strip()]
        )

        # Transformer "animals" en liste
        df["animal_list"] = df["animals"].apply(
            lambda x: [a.strip() for a in str(x).split(";") if a.strip()]
        )

        # Sentiment (peut ne pas exister sur d'anciens fichiers)
        if "sentiment" in df.columns:
            df["sentiment_category"] = pd.cut(
                df["sentiment"],
                bins=[-1.0, -0.05, 0.05, 1.0],
                labels=["Négatif", "Neutre", "Positif"],
            )

        # Traitement des dates (datetime directement pour éviter les types mixtes)
        if "publication_date_detected" in df.columns:
            df["publication_date_detected"] = pd.to_datetime(
                df["publication_date_detected"], errors="coerce"
            )

        return df
    
    except FileNotFoundError:
        st.error("Fichier 'dataset_scraping_nlp.csv' non trouvé. Veuillez vérifier le chemin du fichier.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return pd.DataFrame()

df = load_data()

# Gestion du statut de scraping (compatibilité si colonne absente)
if "scrape_status" not in df.columns:
    df["scrape_status"] = "ok"

total_urls = len(df)
failed_scrapes = len(df[df["scrape_status"] == "scrape_failed"])
too_short = len(df[df["scrape_status"] == "too_short"])

# On ne garde que les articles valides pour les analyses détaillées
df = df[df["scrape_status"] == "ok"]

# Vérifier si les données valides sont chargées
if df.empty:
    st.warning("Aucun article valide à afficher (tous les scrapings ont échoué ou sont trop courts).")
    st.stop()

# ============================================================
# SIDEBAR FILTRES
# ============================================================
st.sidebar.title("🔍 Filtres")

# Filtre par pays
available_countries = sorted([c for c in df["country"].unique() if c != "Unknown"])
selected_country = st.sidebar.selectbox("Pays :", ["Tous"] + available_countries)

# Filtre par langue (filtre global)
available_languages = sorted(df["language"].unique())
selected_language = st.sidebar.selectbox("Langue (filtre global) :", ["Toutes"] + available_languages)

# Langue utilisée pour l'analyse détaillée (maladies, pays, animaux, temps)
analysis_language = st.sidebar.selectbox(
    "Langue pour l'analyse détaillée :", ["Toutes"] + available_languages
)

# Filtre par source
available_sources = sorted([s for s in df["source_nlp"].unique() if s != "Unknown"])
selected_source = st.sidebar.selectbox("Source :", ["Toutes"] + available_sources)

# Filtre par maladie
all_diseases = (
    sorted({d for dlist in df["disease_list"] for d in dlist})
    if "disease_list" in df.columns
    else []
)
selected_diseases = st.sidebar.multiselect(
    "Maladies :", options=all_diseases, default=[]
)

# Filtre par sentiment (si disponible)
if "sentiment_category" in df.columns:
    sentiment_options = ["Tous"] + list(df["sentiment_category"].cat.categories)
    selected_sentiment = st.sidebar.selectbox("Sentiment :", sentiment_options)
else:
    selected_sentiment = "Tous"

# Filtre par date
if (
    "publication_date_detected" in df.columns
    and not df["publication_date_detected"].isna().all()
):
    # Assurer un type datetime pour min/max
    date_series = pd.to_datetime(df["publication_date_detected"], errors="coerce")
    min_date = date_series.min()
    max_date = date_series.max()

    if pd.notna(min_date) and pd.notna(max_date):
        date_range = st.sidebar.date_input(
            "Période :",
            [min_date.date(), max_date.date()],
            min_value=min_date.date(),
            max_value=max_date.date(),
        )

        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date.date(), max_date.date()
    else:
        start_date, end_date = None, None
else:
    start_date, end_date = None, None

# Application des filtres globaux
filtered_df = df.copy()

if selected_country != "Tous":
    filtered_df = filtered_df[filtered_df["country"] == selected_country]
if selected_language != "Toutes":
    filtered_df = filtered_df[filtered_df["language"] == selected_language]
if selected_source != "Toutes":
    filtered_df = filtered_df[filtered_df["source_nlp"] == selected_source]
if selected_diseases and "disease_list" in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df["disease_list"].apply(
            lambda lst: any(d in lst for d in selected_diseases)
        )
    ]
if selected_sentiment != "Tous" and "sentiment_category" in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df["sentiment_category"] == selected_sentiment
    ]
if start_date and end_date and "publication_date_detected" in filtered_df.columns:
    # Reconvertir proprement en datetime pour la comparaison
    date_col = pd.to_datetime(
        filtered_df["publication_date_detected"], errors="coerce"
    ).dt.date
    mask = (date_col >= start_date) & (date_col <= end_date)
    filtered_df = filtered_df[mask]

# Sous-ensemble dédié à l'analyse détaillée (sans perdre les autres langues)
analysis_df = filtered_df.copy()
if analysis_language != "Toutes":
    analysis_df = analysis_df[analysis_df["language"] == analysis_language]

# ============================================================
# KPI
# ============================================================
st.markdown('<div class="section-header">📈 Indicateurs Clés</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{len(filtered_df)}</div>
        <div class="kpi-label">Articles valides</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{total_urls}</div>
        <div class="kpi-label">URLs totales</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{failed_scrapes}</div>
        <div class="kpi-label">Scraping échoué</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{too_short}</div>
        <div class="kpi-label">Articles trop courts</div>
    </div>
    """, unsafe_allow_html=True)

# Ligne de KPI secondaires (mots, pays, sources)
col5, col6, col7, col8 = st.columns(4)

with col5:
    avg_words = filtered_df['word_count'].mean() if not filtered_df.empty else 0
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{avg_words:.0f}</div>
        <div class="kpi-label">Mots (moyenne)</div>
    </div>
    """, unsafe_allow_html=True)

with col6:
    unique_countries = filtered_df["country"].nunique() if not filtered_df.empty else 0
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{unique_countries}</div>
        <div class="kpi-label">Pays</div>
    </div>
    """, unsafe_allow_html=True)

with col7:
    unique_sources = filtered_df["source_nlp"].nunique() if not filtered_df.empty else 0
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{unique_sources}</div>
        <div class="kpi-label">Sources</div>
    </div>
    """, unsafe_allow_html=True)

with col8:
    articles_with_diseases = len(filtered_df[filtered_df['disease_list'].str.len() > 0]) if not filtered_df.empty else 0
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{articles_with_diseases}</div>
        <div class="kpi-label">Articles avec maladies</div>
    </div>
    """, unsafe_allow_html=True)

# Statistiques supplémentaires
if not filtered_df.empty:
    col9, col10, col11 = st.columns(3)
    
    with col9:
        total_diseases = sum(filtered_df['disease_list'].str.len())
        st.metric("Mentions de maladies", total_diseases)
    
    with col10:
        detection_rate = (articles_with_diseases / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
        st.metric("Taux de détection", f"{detection_rate:.1f}%")
    
    with col11:
        st.metric("Articles analysés (NLP)", len(filtered_df))

# ============================================================
# VISUALISATIONS
# ============================================================

if not filtered_df.empty:
    # --- Distribution par pays ---
    st.markdown('<div class="section-header">🌍 Distribution par Pays</div>', unsafe_allow_html=True)
    country_counts = analysis_df["country"].value_counts()
    
    if len(country_counts) > 0:
        fig_country = px.bar(
            x=country_counts.index,
            y=country_counts.values,
            labels={"x": "Pays", "y": "Nombre d'articles"},
            color=country_counts.values,
            color_continuous_scale="Blues",
            title="Nombre d'articles par pays"
        )
        fig_country.update_layout(showlegend=False)
        st.plotly_chart(fig_country, use_container_width=True)
    else:
        st.info("Aucun pays à afficher avec les filtres actuels.")

    # --- Distribution par langue ---
    st.markdown('<div class="section-header">🗣️ Distribution par Langue</div>', unsafe_allow_html=True)
    lang_counts = filtered_df["language"].value_counts()
    
    col_lang1, col_lang2 = st.columns([2, 1])
    
    with col_lang1:
        fig_lang = px.pie(
            values=lang_counts.values, 
            names=lang_counts.index, 
            hole=0.4,
            title="Distribution des langues"
        )
        st.plotly_chart(fig_lang, use_container_width=True)
    
    with col_lang2:
        st.dataframe(lang_counts, use_container_width=True)

    # --- Distribution nombre de mots ---
    st.markdown('<div class="section-header">📏 Distribution du Nombre de Mots</div>', unsafe_allow_html=True)
    
    col_words1, col_words2 = st.columns([3, 1])
    
    with col_words1:
        fig_words = px.histogram(
            filtered_df, 
            x="word_count", 
            nbins=20,
            title="Distribution de la longueur des articles",
            labels={"word_count": "Nombre de mots"}
        )
        st.plotly_chart(fig_words, use_container_width=True)
    
    with col_words2:
        word_stats = filtered_df['word_count'].describe()
        st.metric("Moyenne", f"{word_stats['mean']:.0f}")
        st.metric("Médiane", f"{word_stats['50%']:.0f}")
        st.metric("Max", f"{word_stats['max']:.0f}")

    # --- Analyse des maladies ---
    st.markdown('<div class="section-header">🦠 Analyse des Maladies</div>', unsafe_allow_html=True)
    
    # Compter toutes les maladies
    disease_all = []
    for dlist in analysis_df["disease_list"]:
        disease_all.extend(dlist)
    
    if len(disease_all) > 0:
        disease_counts = Counter(disease_all)
        
        # Top 10 des maladies
        top_diseases = dict(disease_counts.most_common(10))
        
        col_disease1, col_disease2 = st.columns(2)
        
        with col_disease1:
            fig_disease_bar = px.bar(
                x=list(top_diseases.keys()),
                y=list(top_diseases.values()),
                title="Top 10 des Maladies Mentionnées",
                labels={"x": "Maladie", "y": "Nombre de mentions"},
                color=list(top_diseases.values()),
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_disease_bar, use_container_width=True)
        
        with col_disease2:
            # Pie chart pour les maladies
            if len(disease_counts) <= 15:  # Éviter les pie charts trop chargés
                fig_disease_pie = px.pie(
                    values=list(disease_counts.values()),
                    names=list(disease_counts.keys()),
                    title="Répartition des Maladies",
                    hole=0.3
                )
                st.plotly_chart(fig_disease_pie, use_container_width=True)
            else:
                # Afficher un tableau à la place
                disease_df = pd.DataFrame(disease_counts.items(), columns=["Maladie", "Fréquence"]).sort_values("Fréquence", ascending=False)
                st.dataframe(disease_df.head(15), use_container_width=True)
        
        # Heatmap maladies par pays
        st.markdown("#### 📍 Maladies par Pays")
        disease_country_data = []
        for _, row in analysis_df.iterrows():
            for disease in row['disease_list']:
                disease_country_data.append({
                    'disease': disease,
                    'country': row['country']
                })
        
        if disease_country_data:
            disease_country_df = pd.DataFrame(disease_country_data)
            heatmap_data = disease_country_df.groupby(['country', 'disease']).size().unstack(fill_value=0)
            
            if not heatmap_data.empty and len(heatmap_data) > 1:
                fig_heat = px.imshow(
                    heatmap_data,
                    title="Distribution des Maladies par Pays",
                    aspect="auto",
                    color_continuous_scale="Blues"
                )
                st.plotly_chart(fig_heat, use_container_width=True)
    
    else:
        st.info("Aucune maladie détectée dans les articles filtrés.")

    # --- Analyse des animaux ---
    st.markdown('<div class="section-header">🐾 Animaux Mentionnés</div>', unsafe_allow_html=True)
    
    animal_all = []
    for alist in analysis_df["animal_list"]:
        animal_all.extend(alist)
    
    if len(animal_all) > 0:
        animal_counts = Counter(animal_all)
        animal_df = pd.DataFrame(animal_counts.items(), columns=["Animal", "Fréquence"]).sort_values("Fréquence", ascending=False)
        
        col_animal1, col_animal2 = st.columns([2, 1])
        
        with col_animal1:
            fig_animal = px.bar(
                animal_df.head(10),
                x="Animal",
                y="Fréquence",
                title="Top 10 des Animaux Mentionnés",
                color="Fréquence",
                color_continuous_scale="Greens"
            )
            st.plotly_chart(fig_animal, use_container_width=True)
        
        with col_animal2:
            st.dataframe(animal_df, use_container_width=True)
    else:
        st.info("Aucun animal détecté dans les articles filtrés.")

    # --- Évolution temporelle ---
    if 'publication_date_detected' in analysis_df.columns and not analysis_df['publication_date_detected'].isna().all():
        st.markdown('<div class="section-header">📅 Évolution Temporelle</div>', unsafe_allow_html=True)
        
        timeline_data = analysis_df.groupby('publication_date_detected').size().reset_index()
        timeline_data.columns = ['Date', 'Nombre d\'articles']
        timeline_data = timeline_data.sort_values('Date')
        
        fig_timeline = px.line(
            timeline_data,
            x='Date',
            y='Nombre d\'articles',
            title='Évolution du nombre d\'articles dans le temps',
            markers=True
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

else:
    st.warning("Aucune donnée à afficher avec les filtres sélectionnés.")

# ============================================================
# TABLEAU DES DONNÉES
# ============================================================
st.markdown('<div class="section-header">📋 Détails des Articles</div>', unsafe_allow_html=True)

if not filtered_df.empty:
    if st.checkbox("Afficher les données détaillées"):
        st.subheader("📊 Données Filtrees")
        
        # Options d'affichage
        col_display1, col_display2 = st.columns(2)
        
        with col_display1:
            show_columns = st.multiselect(
                "Colonnes à afficher :",
                options=["title", "country", "language", "source_nlp", "word_count", "diseases", "animals", "url"],
                default=["title", "country", "language", "word_count", "diseases"]
            )
        
        with col_display2:
            rows_to_show = st.slider("Nombre de lignes à afficher :", 5, 100, 20)
        
        # Recherche
        search_term = st.text_input("🔍 Rechercher dans les titres :")
        
        # Préparation des données
        display_df = filtered_df[show_columns].copy() if show_columns else filtered_df.copy()

        # Formater les colonnes de liste (sans eval)
        if "diseases" in display_df.columns and "disease_list" in filtered_df.columns:
            display_df["diseases"] = filtered_df["disease_list"].apply(
                lambda lst: ", ".join(lst) if lst else "Aucune"
            )
        if "animals" in display_df.columns and "animal_list" in filtered_df.columns:
            display_df["animals"] = filtered_df["animal_list"].apply(
                lambda lst: ", ".join(lst) if lst else "Aucun"
            )
        
        # Appliquer la recherche
        if search_term:
            display_df = display_df[display_df['title'].str.contains(search_term, case=False, na=False)]
        
        # Afficher le tableau
        st.dataframe(display_df.head(rows_to_show), use_container_width=True, height=400)
        
        # Statistiques d'export
        st.info(f"📄 {len(display_df)} articles correspondants aux critères")
        
        # Option d'export
        if st.button("📥 Exporter les données filtrées en CSV"):
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="Télécharger CSV",
                data=csv,
                file_name="donnees_sante_animale_filtrees.csv",
                mime="text/csv"
            )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        Dashboard créé avec <strong>Streamlit</strong> | Données extraites via <strong>NLP</strong> | 
        <em>Dernière mise à jour: {}</em>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M")),
    unsafe_allow_html=True
)