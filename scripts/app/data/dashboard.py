import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from wordcloud import WordCloud
import plotly.express as px

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Analytics - Données Santé Animale",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .kpi-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .section-header {
        font-size: 1.5rem;
        color: #1f77b4;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<div class="main-header">📊 Dashboard Analytique - Données Santé Animale</div>', unsafe_allow_html=True)

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv('dataset_scraping_nlp.csv')
    
    # Nettoyage des données
    df['word_count'] = pd.to_numeric(df['word_count'], errors='coerce').fillna(0)
    df['country'] = df['country'].fillna('Unknown')
    df['language'] = df['language'].fillna('Unknown')
    df['media'] = df['media'].fillna('Unknown')
    
    # Extraction des maladies
    def extract_diseases_from_text(text):
        if pd.isna(text):
            return []
        common_diseases = [
            'avian flu', 'bird flu', 'west nile', 'anthrax', 'newcastle', 
            'foot and mouth', 'rabies', 'monkeypox', 'strangles', 'encephalitis',
            'influenza', 'h5n1', 'h9n2', 'congo fever', 'crimean-congo',
            'equine', 'brucellosis', 'tuberculosis', 'salmonella'
        ]
        found_diseases = []
        text_lower = str(text).lower()
        for disease in common_diseases:
            if disease in text_lower:
                found_diseases.append(disease)
        return found_diseases
    
    df['extracted_diseases'] = df['text'].apply(extract_diseases_from_text)
    
    return df

df = load_data()

# Sidebar pour les filtres
st.sidebar.title("🔍 Filtres")

# Filtre par pays
countries = ['Tous'] + sorted(df['country'].unique().tolist())
selected_country = st.sidebar.selectbox("Sélectionnez un pays:", countries)

# Filtre par langue
languages = ['Toutes'] + sorted(df['language'].unique().tolist())
selected_language = st.sidebar.selectbox("Sélectionnez une langue:", languages)

# Filtre par média
medias = ['Tous'] + sorted(df['media'].unique().tolist())
selected_media = st.sidebar.selectbox("Sélectionnez un média:", medias)

# Application des filtres
filtered_df = df.copy()
if selected_country != 'Tous':
    filtered_df = filtered_df[filtered_df['country'] == selected_country]
if selected_language != 'Toutes':
    filtered_df = filtered_df[filtered_df['language'] == selected_language]
if selected_media != 'Tous':
    filtered_df = filtered_df[filtered_df['media'] == selected_media]

# Section KPI principaux
st.markdown('<div class="section-header">📈 Indicateurs Clés de Performance</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_articles = len(filtered_df)
    st.metric("Nombre total d'articles", total_articles)

with col2:
    avg_word_count = filtered_df['word_count'].mean()
    st.metric("Longueur moyenne des articles", f"{avg_word_count:.0f} mots")

with col3:
    countries_covered = filtered_df['country'].nunique()
    st.metric("Pays couverts", countries_covered)

with col4:
    active_medias = filtered_df['media'].nunique()
    st.metric("Médias actifs", active_medias)

# Première ligne de visualisations
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-header">🌍 Distribution par Pays</div>', unsafe_allow_html=True)
    country_counts = filtered_df['country'].value_counts().head(10)
    fig = px.bar(
        country_counts, 
        x=country_counts.index, 
        y=country_counts.values,
        labels={'x': 'Pays', 'y': "Nombre d'articles"},
        color=country_counts.values,
        color_continuous_scale='blues'
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown('<div class="section-header">🗣️ Distribution par Langue</div>', unsafe_allow_html=True)
    language_counts = filtered_df['language'].value_counts()
    fig = px.pie(
        values=language_counts.values,
        names=language_counts.index,
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)

# Deuxième ligne de visualisations
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-header">📏 Distribution de la Longueur des Articles</div>', unsafe_allow_html=True)
    fig = px.histogram(
        filtered_df, 
        x='word_count',
        nbins=20,
        labels={'word_count': 'Nombre de mots'},
        color_discrete_sequence=['#1f77b4']
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown('<div class="section-header">📰 Longueur Moyenne par Média</div>', unsafe_allow_html=True)
    media_word_counts = filtered_df.groupby('media')['word_count'].mean().sort_values(ascending=False).head(10)
    fig = px.bar(
        media_word_counts, 
        x=media_word_counts.index, 
        y=media_word_counts.values,
        labels={'x': 'Média', 'y': 'Moyenne des mots'},
        color=media_word_counts.values,
        color_continuous_scale='viridis'
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# Analyse des mots-clés et maladies
st.markdown('<div class="section-header">🔤 Analyse des Mots-clés et Maladies</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.subheader("Mots-clés les plus fréquents")
    all_keywords = []
    for keywords in filtered_df['keywords'].dropna():
        if isinstance(keywords, str):
            keyword_list = [kw.strip().lower() for kw in keywords.split(',') if kw.strip()]
            all_keywords.extend(keyword_list)
    if all_keywords:
        keyword_counts = Counter(all_keywords).most_common(15)
        keywords_df = pd.DataFrame(keyword_counts, columns=['Mot-clé', 'Fréquence'])
        fig = px.bar(
            keywords_df, 
            x='Fréquence', 
            y='Mot-clé',
            orientation='h',
            color='Fréquence',
            color_continuous_scale='teal'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aucun mot-clé disponible dans les données filtrées")

with col2:
    st.subheader("Maladies mentionnées")
    all_diseases = []
    for diseases in filtered_df['extracted_diseases']:
        all_diseases.extend(diseases)
    if all_diseases:
        disease_counts = Counter(all_diseases).most_common(10)
        diseases_df = pd.DataFrame(disease_counts, columns=['Maladie', 'Fréquence'])
        fig = px.pie(
            diseases_df, 
            values='Fréquence', 
            names='Maladie',
            hole=0.3
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aucune maladie détectée dans les données filtrées")

# Analyse temporelle
st.markdown('<div class="section-header">📅 Analyse Temporelle</div>', unsafe_allow_html=True)

# Dates fictives pour l'analyse
filtered_df['date'] = pd.date_range(start='2024-01-01', periods=len(filtered_df), freq='D')
monthly_counts = filtered_df.groupby(filtered_df['date'].dt.to_period('M')).size()
monthly_counts.index = monthly_counts.index.astype(str)

fig = px.line(
    x=monthly_counts.index,
    y=monthly_counts.values,
    labels={'x': 'Mois', 'y': "Nombre d'articles"},
    markers=True
)
fig.update_traces(line=dict(color='#ff7f0e', width=3))
st.plotly_chart(fig, use_container_width=True)

# Tableau des données détaillées
st.markdown('<div class="section-header">📋 Données Détailées</div>', unsafe_allow_html=True)
if st.checkbox("Afficher les données détaillées"):
    st.dataframe(
        filtered_df[['code', 'media', 'country', 'language', 'title', 'word_count']],
        use_container_width=True
    )

# Statistiques supplémentaires
st.markdown('<div class="section-header">📊 Statistiques Supplémentaires</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Articles avec plus de 200 mots", 
        f"{len(filtered_df[filtered_df['word_count'] > 200])} ({len(filtered_df[filtered_df['word_count'] > 200])/len(filtered_df)*100:.1f}%)"
    )

with col2:
    articles_with_keywords = filtered_df['keywords'].notna().sum()
    st.metric(
        "Articles avec mots-clés", 
        f"{articles_with_keywords} ({articles_with_keywords/len(filtered_df)*100:.1f}%)"
    )

with col3:
    articles_with_diseases = filtered_df[filtered_df['extracted_diseases'].apply(len) > 0].shape[0]
    st.metric(
        "Articles mentionnant des maladies", 
        f"{articles_with_diseases} ({articles_with_diseases/len(filtered_df)*100:.1f}%)"
    )

# Footer
st.markdown("---")
st.markdown(
    "**Dashboard créé avec Streamlit** | "
    "Données: Articles sur la santé animale | "
    "Dernière mise à jour: Données chargées depuis le fichier CSV"
)
