from fastapi import FastAPI
from .dtos import Usuario
from . import recomendation_service
from contextlib import asynccontextmanager
from . import data_service


@asynccontextmanager
async def get_dependencies(app: FastAPI):
    data_service.get_dependencies()
    yield

app = FastAPI(lifespan=get_dependencies)


@app.post("/recomendar")
def recomendar(usuario: Usuario):    
    return recomendation_service.recomendar(usuario, n_recomendations=10)
