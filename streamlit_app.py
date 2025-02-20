import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity


df1 = pd.read_csv('archive/anime.csv')

df1['genre'] = df1['genre'].fillna('Unknown')
df1['type'] = df1['type'].fillna('Unknown')
df1['rating'] = df1['rating'].fillna(df1['rating'].median())

df1['genre'] = df1['genre'].apply(lambda x: [g.strip() for g in x.split(',')])

mlb = MultiLabelBinarizer()
genre_encoded = pd.DataFrame(mlb.fit_transform(df1['genre']), columns=mlb.classes_)

df1 = pd.concat([df1, genre_encoded], axis=1)
df1.drop('genre', axis=1, inplace=True)

df1 = df1[df1['episodes'] != 'Unknown']
df1['episodes'] = pd.to_numeric(df1['episodes'], errors='coerce').fillna(0).astype(int)

similarity_matrix = cosine_similarity(df1.iloc[:, 3:])

all_genres = sorted(list(mlb.classes_))

def create_anime_vector(episodes, genres, rating):
    columns = df1.columns[3:]
    new_anime = pd.DataFrame([[0]*len(columns)], columns=columns)
    
    for g in genres:
        if g in new_anime.columns:
            new_anime[g] = 1
            
    new_anime['episodes'] = int(episodes)
    new_anime['rating'] = float(rating)
    
    return new_anime


def recommend_anime(genres, episodes, rating, n=5):
    anime_vector = create_anime_vector(episodes, genres, rating)
    similarity = cosine_similarity(anime_vector, df1.iloc[:, 3:].values)
    
    scores = list(enumerate(similarity[0]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:n]
    
    recommended_anime = [df1.iloc[i[0]]['name'] for i in scores]
    
    return recommended_anime



st.title("Аниме-рекомендательная система")


selected_genres = st.multiselect("Выбери жанры", all_genres)


episodes_input = st.number_input("Количество эпизодов", min_value=0, max_value=5000, value=12)


rating_input = st.slider("Рейтинг аниме (0-10)", 0.0, 10.0, 5.0, 0.1)

if st.button("Рекомендовать"):
    recs = recommend_anime(selected_genres, episodes_input, rating_input, n=5)
    st.write("-----------------------------------------")
    st.subheader(":rainbow[**Тебе могут понравиться:**]")
    for r in recs:
        st.write(f"- {r}")