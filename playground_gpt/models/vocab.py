from pydantic import BaseModel


class Vocab(BaseModel):
    token: str
    token_id: int
