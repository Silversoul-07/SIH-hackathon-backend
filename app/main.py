from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from app.route import router

app = FastAPI()
app.include_router(router)

@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")