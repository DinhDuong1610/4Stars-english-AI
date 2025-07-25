from pydantic import BaseModel
from typing import List

class DiffResult(BaseModel):
    type: str  # 'equal', 'insert', 'delete'
    text: str

class DictationCheckRequest(BaseModel):
    user_text: str
    correct_text: str

class DictationCheckResponse(BaseModel):
    score: float
    diffs: List[DiffResult]