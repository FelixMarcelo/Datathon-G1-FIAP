from fastapi import FastAPI
from fastapi.responses import JSONResponse
from .dtos import Usuario
from . import recomendation_service
from contextlib import asynccontextmanager
from . import data_service


data_service.get_dependencies()

app = FastAPI()


@app.post("/recomendar")
def recomendar(usuario: Usuario):    
    res = recomendation_service.recomendar(usuario, n_recomendations=10)
    return JSONResponse(content=res)
