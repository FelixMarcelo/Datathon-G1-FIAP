import joblib
import pandas as pd
import numpy as np
import glob

import pandas as pd
import numpy as np


def combine_text(df):
    return df["title"] + " " + df["caption"] + " " + df["body"]


def limpar_valor(valor):
    valor_limpo = (valor.
                replace('\n', ' ').
                replace("'", ' ').
                replace("[", ' ').
                replace("]", ' ').
                replace(',', ' ').
                strip().split()
        ) 
    
    return valor_limpo


def tratar_base_treino(df, df_itens, model, vectorizer):
    df = df.set_index("userId").to_dict(orient="index")
    df_users = pd.DataFrame()
    n_iteractions = 0
    total_iteractions = len(df)
    df_itens_indexed = df_itens.set_index("page").to_dict(orient="index")
    for key in df:
        row = df[key]
        user = key
        print(f'user: {user}')
        hist = limpar_valor(row['history'])
        print(f'hist: {hist}')
        
        timestampHistory = list(map(int, limpar_valor(row['timestampHistory'])))
        print(f'timestampHistory: {timestampHistory}')
        
        numberOfClicksHistory = list(map(int, limpar_valor(row['numberOfClicksHistory'])))
        print(f'numberOfClicksHistory: {numberOfClicksHistory}')
        
        timeOnPageHistory = list(map(int, limpar_valor(row['timeOnPageHistory'])))
        print(f'timeOnPageHistory: {timeOnPageHistory}')
        
        scrollPercentageHistory = list(map(float, limpar_valor(row['scrollPercentageHistory'])))
        print(f'scrollPercentageHistory: {scrollPercentageHistory}')
        
        pageVisitsCountHistory = list(map(int, limpar_valor(row['pageVisitsCountHistory'])))
        print(f'pageVisitsCountHistory: {pageVisitsCountHistory}')
        
        timestampHistory_max = max(timestampHistory)
        
        numberOfClicksHistory_mean = np.mean(numberOfClicksHistory)
        numberOfClicksHistory_std = max(np.std(numberOfClicksHistory), 1)
        
        timeOnPageHistory_mean = np.mean(timeOnPageHistory)
        timeOnPageHistory_std = max(np.std(timeOnPageHistory), 1)
        
        scrollPercentageHistory_mean = np.mean(scrollPercentageHistory)
        scrollPercentageHistory_std = max(np.std(scrollPercentageHistory), 1)
        
        pageVisitsCountHistory_mean = np.mean(pageVisitsCountHistory)
        pageVisitsCountHistory_std = max(np.std(pageVisitsCountHistory), 1)
        
        user_dict = {'userId': user, **{f'{i}': 0 for i in model.labels_}}
        for n_item in range(len(hist)):
            timestampHistory_score = timestampHistory[n_item] / timestampHistory_max
            numberOfClicksHistory_score = (numberOfClicksHistory[n_item] - numberOfClicksHistory_mean) / numberOfClicksHistory_std
            timeOnPageHistory_score = (timeOnPageHistory[n_item] - timeOnPageHistory_mean) / timeOnPageHistory_std
            scrollPercentageHistory_score = (scrollPercentageHistory[n_item] - scrollPercentageHistory_mean) / scrollPercentageHistory_std
            pageVisitsCountHistory_score = (pageVisitsCountHistory[n_item] - pageVisitsCountHistory_mean) / pageVisitsCountHistory_std
            
            final_item_score = timestampHistory_score * 1.3 + numberOfClicksHistory_score * 1 + timeOnPageHistory_score * 1 + scrollPercentageHistory_score * 1.2 + pageVisitsCountHistory_score * 1
            item_row = df_itens_indexed.get(hist[n_item], None)
            if len(item_row) == 0 or item_row == None:
                print(" --- Item não encontrado na base de dados --- ")
                continue
            X = vectorizer.transform([combine_text(item_row)])
            cluster = model.predict(X)[0]
            
            user_dict[f'{cluster}'] += final_item_score
            
        df_users = pd.concat([df_users, pd.DataFrame([user_dict])], ignore_index=True)
        
        # Porcentagem de progresso
        n_iteractions += 1        
        progresso = (n_iteractions / total_iteractions) * 100
        print(f"Processando: {progresso:.2f}% ", end="\r")
        
    return df_users, hist


def dict_recomendations(df_user):
    sum = df_user.iloc[0, 1:].sum()
    n_recomendations = 10
    user_dict = {f'{df_user.columns[n]}': round((df_user[df_user.columns[n]].values[0] / sum) * n_recomendations) for n in range(len(df_user.columns)) if df_user.columns[n] != "userId"}
    return dict(sorted(user_dict.items(), key=lambda item: item[1], reverse=True))


def recomend(user_dict, user_hist, last_news, n_recomendations = 10):
    n = 0
    n_recomendations = n_recomendations
    df = pd.DataFrame()
    for key in user_dict:
        value = user_dict[key]
        if value <= 0:
            continue
        
        if value >= n_recomendations:
            value = n_recomendations
            
        
        if n + value <= n_recomendations:
            n += value
            temp = last_news[last_news['cluster'] == int(key)].head(value)
            df = pd.concat([df, temp], ignore_index=True)        
            
            continue
            
        
        diff = n_recomendations - n
        
        if diff > 0:
            first_key = next(iter(user_dict))
            temp = last_news[last_news['cluster'] == int(first_key)].head(diff)
            df = pd.concat([df, temp], ignore_index=True)
        
    if len(df) < n_recomendations:
        df = pd.concat([df, last_news[~last_news['page'].isin(user_hist)].sort_values(by='issued').head(n_recomendations)])
    return df.head(n_recomendations)