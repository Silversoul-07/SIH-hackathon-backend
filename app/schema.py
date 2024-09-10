from pydantic import BaseModel

class prediction(BaseModel):
    class_name: str
    confidence_score: float