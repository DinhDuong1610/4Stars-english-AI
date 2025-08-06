from fastapi import APIRouter

from app.services.dictation_service import check_dictation_answer
from app.schemas.dictation_schema import DictationCheckRequest, DictationCheckResponse

router = APIRouter()

@router.post("/check", response_model=DictationCheckResponse)
def check_dictation(request: DictationCheckRequest):

    result = check_dictation_answer(request.user_text, request.correct_text)

    return result