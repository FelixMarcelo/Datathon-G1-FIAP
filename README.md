# Datathon - Sistema de Recomendação
## Case Globo - G1


### Construção do modelo

**Como lidar com o Cold Start?** 

Usuários ou itens com pouca informação receberão como recomendação as 10 primeiras notícias de um ranking que considera data (recência) e número de clicks como critério de classificação.

**Item 1: Treinamento**

Vetorizar o título das últimas 10 notícias que o usuário clicou e classificá-los entre N grupos. 

A vetorização dos títulos será feita utilizando a classe **TfidfVectorizer** do **sklearn**.

A clusterização será feita utilizando algorítmo **KMeans**. 

Exemplo de uso:

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 1️⃣ Existing dataset (Users & Last 10 Movies)
data = {
    "userId": [101, 102, 103, 104, 105],
    "movies_watched": [
        "Die Hard, Avengers, Mad Max, Inception, Gladiator, John Wick, Batman, Deadpool, Iron Man, Logan",
        "Titanic, The Notebook, P.S. I Love You, La La Land, Pride & Prejudice, Me Before You, A Walk to Remember, Romeo + Juliet, The Fault in Our Stars, Love Actually",
        "Interstellar, Inception, The Matrix, Star Wars, Blade Runner, Arrival, Gravity, The Martian, 2001: A Space Odyssey, Ad Astra",
        "The Conjuring, Insidious, Paranormal Activity, The Exorcist, Sinister, Hereditary, The Ring, It, Halloween, A Nightmare on Elm Street",
        "Toy Story, Finding Nemo, Up, Shrek, Frozen, The Lion King, Aladdin, Moana, Tangled, Beauty and the Beast"
    ]
}

df = pd.DataFrame(data)
df["movies_text"] = df["movies_watched"].str.replace(",", " ")  # Replace commas with spaces

# 2️⃣ Train the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["movies_text"])  # Convert movie text into numerical features

# 3️⃣ Train K-Means Clustering
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df["cluster"] = kmeans.fit_predict(X)

# 4️⃣ Classifying a New User's Movie History
def classify_user(new_movies):
    # Convert movie list into a single text string
    new_movies_text = " ".join(new_movies)
    
    # Transform using the trained TF-IDF Vectorizer
    new_movies_vector = vectorizer.transform([new_movies_text])
    
    # Predict the cluster
    predicted_cluster = kmeans.predict(new_movies_vector)[0]
    
    return predicted_cluster

# 5️⃣ Example: Classify a new user
new_user_movies = [
    "The Godfather", "Goodfellas", "Pulp Fiction", "Scarface", 
    "The Departed", "Reservoir Dogs", "Casino", "Heat", "American Gangster", "The Irishman"
]

user_cluster = classify_user(new_user_movies)
print(f"New User is classified into Cluster {user_cluster}")
```

A partir da classificação do usuário, recomendar as 10 notícias mais bem classificadas do seu cluster, considerando novamente os critérios de recência e número de cliques, excluindo dessa lista as notícias já lidas pelo usuário.

Obs: Se não estivermos satisfeitos com o desempenho do modelo neste ponto, podemos utilizar esta nova base construida com o KMeans para treinar um outro algorítmo de classificação.

**Item 2: Salvamento do modelo**

Utilizar o pickle para salvar o modelo.

### Item 4: Criação da API para previsões

Utilizar o FastAPI como framework para construir a API REST.

Endpoints:

/recomendar:

body:
```json
{
    "id_usuario": str,
    "lista_ultimas_noticias": list[str]
}
```

### Item 4: Empacotamento com Docker

### Item 5: Testes e validação da API

### Dúvidas:

Precisamos nos preocupar com a pipeline de dados e retreino do modelo? Ex. Rotinas para inclusão de novas notícias e ações dos usuários. Rotina para retreino do modelo considerando essas novas informações.

