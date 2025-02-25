from fastapi import FastAPI
from .dtos import Usuario


app = FastAPI()

@app.post("/recomendar")
def recomendar(usuario: Usuario):
    return usuario
