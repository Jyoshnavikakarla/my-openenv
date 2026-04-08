from pydantic import BaseModel

class Observation(BaseModel):
    email: str
    sender: str
    step: int

class Action(BaseModel):
    category: str   # billing / technical / general
    priority: str   # low / medium / high
    response: str

class Reward(BaseModel):
    score: float