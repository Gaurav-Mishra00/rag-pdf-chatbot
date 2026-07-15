import logging
import os
from contextlib import asynccontextmanager
import anyio
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.database import get_db_connection, init_db
from app.api.router import api_router

# Setup logger configuration
logging.basicConfig(
    level=logging.getLevelName(settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup event: Ensure data directories exist and init DB
    logger.info("Starting up RAG Chatbot API...")
    os.makedirs(os.path.dirname(settings.FAISS_INDEX_PATH), exist_ok=True)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    init_db()
    logger.info("Required local directories and database verified.")
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

    # Simple healthcheck endpoint (liveness — always 200 if process is running)
    @app.get("/health", tags=["System"])
    async def health_check():
        return {
            "status": "healthy",
            "app_name": settings.APP_NAME,
            "environment": settings.APP_ENV,
        }

    # Readiness probe — checks external dependencies
    @app.get("/health/ready", tags=["System"])
    async def health_ready():
        """
        Readiness probe for load balancers and orchestrators.
        Returns 200 when all dependencies (DB, FAISS index file) are reachable.
        Returns 503 with per-component detail when any dependency is unhealthy.
        """
        components = {}

        # Check 1: SQLite reachability
        def _check_db():
            with get_db_connection() as conn:
                conn.execute("SELECT 1").fetchone()

        try:
            await anyio.to_thread.run_sync(_check_db)
            components["database"] = "ok"
        except Exception as exc:
            logger.error("Readiness check — DB not reachable: %s", exc)
            components["database"] = "error"

        # Check 2: FAISS index file present on disk
        import os as _os
        index_file = _os.path.join(settings.FAISS_INDEX_PATH, "index.faiss")
        if _os.path.exists(index_file):
            components["vector_store"] = "ok"
        else:
            components["vector_store"] = "not_initialized"

        all_ok = all(v == "ok" for v in components.values())
        payload = {
            "status": "ready" if all_ok else "not_ready",
            "components": components,
        }
        http_status = status.HTTP_200_OK if all_ok else status.HTTP_503_SERVICE_UNAVAILABLE
        return JSONResponse(status_code=http_status, content=payload)

    return app


app = create_app()
