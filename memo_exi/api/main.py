"""
FastAPI Application Entrypoint for MEMO-EXI.
"""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from memo_exi.api.routes import router
from memo_exi.config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("memo_exi")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup and shutdown handler."""
    logger.info("Initializing MEMO-EXI storage directories...")
    settings.ensure_directories()
    logger.info("MEMO-EXI API successfully initialized.")
    yield
    logger.info("MEMO-EXI API shutting down.")


app = FastAPI(
    title="MEMO-EXI API",
    description="Adaptive Temporal Memory Management Engine for LLM Agents Enhancement",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(router)


@app.get("/", tags=["Health"])
def health_check():
    """Service health and system status."""
    return {
        "service": "MEMO-EXI",
        "version": "1.0.0",
        "status": "operational",
        "description": "Adaptive Temporal Memory Management for LLM Agents Enhancement",
    }
