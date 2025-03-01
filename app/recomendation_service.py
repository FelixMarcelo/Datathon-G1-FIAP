from .data_service import df_usuarios, df_itens, kmeans, vectorizer, df_last_news
from . import backend
from .dtos import Usuario


def recomendar(usuario: Usuario, n_recomendations: int):
    
    if usuario.usuario_id not in df_usuarios:
        return df_last_news.head(n_recomendations)
    
    df_usuario, usuario_hist = backend.tratar_base_treino(df=df_usuarios[df_usuarios['userId'] == usuario.usuario_id], df_itens=df_itens, model=kmeans, vectorizer=vectorizer)
    
    usuario_dict = backend.dict_recomendations(df_user=df_usuario)
    
    rec = backend.recomend(user_dict=usuario_dict, user_hist=usuario_hist, last_news=df_last_news, n_recomendations=n_recomendations)
    
    return rec.to_dict(orient="index")
    
    