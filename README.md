# Datathon - Sistema de Recomendação
## Case Globo - G1


### Construção do modelo

## Sobre a estratégia: 
Por se tratar de um canal de notícias, optamos por focar principalmente na **data**, no **conteúdo** das notícias e no **perfil** de leitura do usuário. 

A ideia é ser capaz de identificar clusters por meio do conteúdo das notícias (título, descrição e conteúdo) e definir um "score" de engajamento do usuário dado o seu histório de leitura. Este dois fatores, combinados, resultam no perfil do usuário e é através deste perfil que serão feitas as recomendações. 

### Scores de Engajamento
Para definição dos scores de engajamento, foram utilizadas as seguintes colunas: 

**timestampHistory, numberOfClicksHistory, timeOnPageHistory, scrollPercentageHistory e pageVisitsCountHistory.** 

Foram dados pesos a cada uma delas e o cálculo foi baseado na fórmula de padronização das variáveis, da seguinte forma:

```python
timestampHistory_score = timestampHistory[n_item] / timestampHistory_max
numberOfClicksHistory_score = (numberOfClicksHistory[n_item] - numberOfClicksHistory_mean) / numberOfClicksHistory_std
timeOnPageHistory_score = (timeOnPageHistory[n_item] - timeOnPageHistory_mean) / timeOnPageHistory_std
scrollPercentageHistory_score = (scrollPercentageHistory[n_item] - scrollPercentageHistory_mean) / scrollPercentageHistory_std
pageVisitsCountHistory_score = (pageVisitsCountHistory[n_item] - pageVisitsCountHistory_mean) / pageVisitsCountHistory_std

final_item_score = timestampHistory_score * 1.3 + numberOfClicksHistory_score * 1 + timeOnPageHistory_score * 1 + scrollPercentageHistory_score * 1.2 + pageVisitsCountHistory_score * 1
```

As variáveis de tempo e porcentagem de leitura receberam pesos maiores, dada a natureza do negócio. 

Para cada notícia no histórico do usuário, o "score" final de engajamento é definido então como a soma dessas variáveis.


**Como lidar com o Cold Start?** 

Usuários ou itens com pouca ou nenhuma informação receberão como recomendação as 10 primeiras notícias de um ranking que considera data (recência) como critério.

**Item 1: Treinamento**

A vetorização dos títulos, descrição e conteúdo da matéria será feita utilizando a classe **TfidfVectorizer** do **sklearn**.

Utilizou-se a base de itens para treinar o modelo com o algotítmo **Kmeans** e **clusterizar** as notícias em 20 grupos a partir da proximidade do seu conteúdo.

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

Após realizado o treinamento do modelo, foi construída uma pipline de tratamento dos dados de histórico do usuário calculando o seu score de engajamento para cada uma das notícias que já consumiu. 

Essa base foi chamada então de perfil do usuário e é a partir dela que são feitas as recomendações. 

Caso o usuário tenha um histórico na base de dados do G1, o seu perfil será calculado e as recomendações serão feitas preferencialmente a partir dele. Caso este histórico não seja suficiente para a geração de 10 recomendações, os itens faltantes serão indicados considerando a data de lançamento.
**Item 2: Salvamento do modelo**

Utilizou-se a biblioteca joblib para salvar tanto o **modelo** quanto o **vetorizador**.

### Item 3: Criação da API para previsões

Utilizou-se a biblioteca **FastAPI** como framework para construir a API REST.

Endpoints:

/recomendar:

body da requisição POST:
```json
{
    "id_usuario": str
}
```

### Item 4: Empacotamento com Docker
Utilizamos um Dockerfile para empacotar a aplicação e torná-la produtiva

### Item 5: Testes e validação da API
Gravou-se como comprovação da funcionalidade.

# Como rodar a aplicação? 

## Pré requisito: Docker

## Comandos: 
## IMPORTANTE: 
Para que a aplicação funcione corretamente, após clonar o repositório, faça o download desses quatro arquivos e cole-os no diretório: app/model

https://drive.google.com/file/d/1tL02AJ_whHud-KrSFBku5MVw_G6nlK12/view?usp=gmail
https://drive.google.com/file/d/18lDf6CKruX0keH8o-dt3BGgGJi8sj0Ew/view?usp=gmail
https://drive.google.com/file/d/1Fhql_r8pO39UWX1WvQ8HGNKKx0Trm7u1/view?usp=gmail
https://drive.google.com/file/d/1XYRtyPtmO3sRZoatHsMBnT_4jyhoiXs6/view?usp=gmail

Após realizada esta etapa, rode os comandos abaixo

**A partir da raiz do projeto:**
```
docker build -t recomendation-system-g1-fiap .
docker run -p 8000:8000 --name datathon-g1 recomendation-system-g1-fiap
```

