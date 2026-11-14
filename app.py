import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Recomendador de Películas IA", page_icon="🎬", layout="wide")
st.title("🎬 Recomendador de Películas con IA")
st.markdown("Selecciona una película y la IA te recomendará otras similares usando **similitud de coseno** sobre sus géneros.")

@st.cache_data
def load_data():
    movies = pd.read_csv('movies.csv')
    movies['genres_clean'] = movies['genres'].str.replace('|', ' ', regex=False)
    return movies

movies = load_data()

@st.cache_resource
def build_similarity(df):
    cv = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
    count_matrix = cv.fit_transform(df['genres_clean'])
    return cosine_similarity(count_matrix)

cosine_sim = build_similarity(movies)

def get_recommendations(title, n=5):
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    if title not in indices:
        return None
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['title', 'genres']]

st.subheader("Selecciona una película:")
movie_list = movies['title'].head(500).tolist()
selected_movie = st.selectbox("Película que te gusta:", movie_list)

if st.button("🔍 Recomendar películas similares"):
    with st.spinner("Analizando similitudes..."):
        recs = get_recommendations(selected_movie)
        if recs is not None and not recs.empty:
            st.success(f"Basado en **{selected_movie}**, te recomendamos:")
            for _, row in recs.iterrows():
                col1, col2 = st.columns([3, 1])
                col1.write(f"**{row['title']}**")
                col2.caption(row['genres'])

st.sidebar.info("Proyecto de IA: recomendador content-based con similitud de coseno sobre MovieLens.")