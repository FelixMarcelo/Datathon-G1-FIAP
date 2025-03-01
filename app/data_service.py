import pandas as pd
import glob
import joblib
from . import backend
import os
from pathlib import Path
BASE_DIR = Path(__file__).parent.resolve()
print("Current Working Directory:", BASE_DIR)

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
    """Load all item CSV files from the itens directory."""
    itens_path = BASE_DIR / "model/itens"
    lista_itens_csv = glob.glob(str(itens_path / "*.csv"))

    if not lista_itens_csv:
        print("⚠️ Nenhum arquivo CSV encontrado em", itens_path)
        return pd.DataFrame()

    print(f"📂 {len(lista_itens_csv)} arquivos CSV encontrados para itens.")
    return pd.concat([pd.read_csv(file) for file in lista_itens_csv], ignore_index=True)


def get_users():
    """Load user data from CSV."""
    users_path = BASE_DIR / "model/treino_parte1.csv"
    
    if not users_path.exists():
        raise FileNotFoundError(f"🚨 Arquivo não encontrado: {users_path}")

    print(f"📥 Carregando usuários de {users_path}")
    return pd.read_csv(users_path)


def get_models():
    """Load pre-trained models."""
    vectorizer_path = BASE_DIR / "model/vectorizer.pkl"
    kmeans_path = BASE_DIR / "model/kmeans.pkl"

    if not vectorizer_path.exists() or not kmeans_path.exists():
        raise FileNotFoundError("🚨 Modelos não encontrados. Verifique os arquivos vectorizer.pkl e kmeans.pkl")

    print("🔍 Carregando modelos...")
    vectorizer = joblib.load(vectorizer_path)
    kmeans = joblib.load(kmeans_path)

    return kmeans, vectorizer


def get_last_news(kmeans, vectorizer):
    """Load last news and classify using the KMeans model."""
    news_path = BASE_DIR / "model/last_news.csv"

    if not news_path.exists():
        raise FileNotFoundError(f"🚨 Arquivo de notícias não encontrado: {news_path}")

    print(f"📰 Carregando últimas notícias de {news_path}")
    last_news = pd.read_csv(news_path)

    X = vectorizer.transform(backend.combine_text(last_news))
    last_news['cluster'] = kmeans.predict(X)

    return last_news