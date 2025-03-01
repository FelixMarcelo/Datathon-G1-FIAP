from . import data_service
from . import backend
from .dtos import Usuario


def recomendar(usuario: Usuario, n_recomendations: int):
    if usuario.usuario_id not in data_service.df_usuarios['userId'].values:
        print(f'Usuario {usuario.usuario_id} não existe na base de dados')
        return data_service.df_last_news.head(n_recomendations).to_dict(orient="records")
    print(f'Usuario {usuario.usuario_id} presente na base')
    print("Gerando recomendação...")
    
    df_usuario, usuario_hist = backend.tratar_base_treino(df=data_service.df_usuarios[data_service.df_usuarios['userId'] == usuario.usuario_id], df_itens=data_service.df_itens, model=data_service.kmeans, vectorizer=data_service.vectorizer)
    
    usuario_dict = backend.dict_recomendations(df_user=df_usuario)
    
    rec = backend.recomend(user_dict=usuario_dict, user_hist=usuario_hist, last_news=data_service.df_last_news, n_recomendations=n_recomendations)
    return rec.to_dict(orient="records")
    
    