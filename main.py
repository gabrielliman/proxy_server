from fastapi import FastAPI
import uvicorn

from config.settings import ROUTING_MODE, KV_CACHE_THRESHOLD
from api.endpoints_chat import router as chat_router
from api.endpoints_completion import router as completion_router
from api.endpoints_status import router as status_router
from metrics.monitor import start_monitoring_tasks


app = FastAPI()

@app.on_event("startup")
async def startup_event():
    start_monitoring_tasks(app)


app.include_router(chat_router)
app.include_router(completion_router)
app.include_router(status_router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
