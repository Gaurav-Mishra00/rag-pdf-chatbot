import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.api.router import api_router

# Setup logger configuration
logging.basicConfig(
    level=logging.getLevelName(settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup event: Ensure data directories exist
    logger.info("Starting up RAG Chatbot API...")
    os.makedirs(os.path.dirname(settings.FAISS_INDEX_PATH), exist_ok=True)
    logger.info("Required local directories verified.")
    yield
    # Shutdown event
    logger.info("Shutting down RAG Chatbot API...")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        description="A production-ready RAG chatbot backend API using LangChain & FAISS",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Setup CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Adjust for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global Exception Handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception on {request.url.path}: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal server error occurred. Please check logs."},
        )

    # Register endpoints router
    app.include_router(api_router, prefix="/api/v1")

    # Simple healthcheck endpoint
    @app.get("/health", tags=["System"])
    async def health_check():
        return {
            "status": "healthy",
            "app_name": settings.APP_NAME,
            "environment": settings.APP_ENV,
        }

    return app


app = create_app()
