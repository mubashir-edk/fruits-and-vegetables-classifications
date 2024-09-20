from fastapi import FastAPI
from routers import train_model, test
import uvicorn

app = FastAPI()

app.include_router(train_model.router)
app.include_router(test.router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
