from pydantic import BaseModel
from typing import List


class Usuario(BaseModel):
    usuario_id: str
    
