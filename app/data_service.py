import pandas as pd
import glob
import joblib
from . import backend
import os
print("Current Working Directory:", os.getcwd())

df_usuarios = None
df_itens = None
df_last_news = None
kmeans = None
vectorizer = None

def get_dependencies():
    global df_usuarios
    global df_itens
    global kmeans
    global vectorizer
    global df_last_news
    
    print("Importando dependencias...")
    
    df_usuarios = get_users()
    df_itens = get_itens()
    kmeans, vectorizer = get_models()
    df_last_news = get_last_news(kmeans=kmeans, vectorizer=vectorizer)
    
    print("Importação finalizada.")

def get_itens():
    lista_itens_csv = glob.glob(f'{os.getcwd()}/app/resources/itens/itens/*.csv')

    return pd.concat([pd.read_csv(file) for file in lista_itens_csv], ignore_index=True)


def get_users():
    return pd.read_csv(f"{os.getcwd()}/app/resources/files/treino/treino_parte1.csv")


def get_models():
    vectorizer = joblib.load(f'{os.getcwd()}/app/model/vectorizer.pkl')
    kmeans = joblib.load(f'{os.getcwd()}/app/model/kmeans.pkl')
    
    return kmeans, vectorizer


def get_last_news(kmeans, vectorizer):
    last_news = pd.read_csv(f'{os.getcwd()}/app/model/last_news.csv')
    X = vectorizer.transform(backend.combine_text(last_news))
    last_news['cluster'] = kmeans.predict(X)
    
    return last_news