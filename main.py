from fastapi import FastAPI
from app.api.v1.routers import dictation as dictation_router

app = FastAPI(title="Dictation API")

app.include_router(
    dictation_router.router,
    prefix="/api/v1/dictation",
    tags=["Dictation"]
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Dictation API!"}