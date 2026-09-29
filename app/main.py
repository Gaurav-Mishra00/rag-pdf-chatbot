import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
import anyio
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html, get_swagger_ui_oauth2_redirect_html
from fastapi.staticfiles import StaticFiles

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

    # Self-healing storage reconciliation on startup
    try:
        from app.api.deps import get_vector_store
        from app.core.database import reconcile_storage_layers
        vector_store = get_vector_store()
        reconcile_storage_layers(vector_store, settings.UPLOAD_DIR)
    except Exception as e:
        logger.error("Failed to run startup storage reconciliation: %s", e)

    # Warn if using default API key in non-testing environments
    if settings.API_KEY == "change_me_in_production" and settings.APP_ENV != "testing":
        logger.warning(
            "SECURITY WARNING: The API_KEY is configured with the default value 'change_me_in_production'. "
            "Please configure a unique secure API_KEY in production."
        )

    # Automated snapshot backup of data assets on startup
    try:
        from app.core.backup import backup_data_assets
        backup_data_assets()
    except Exception as e:
        logger.error("Failed to run automated startup database backup: %s", e)

    logger.info("Required local directories and database verified.")
    yield
    # Shutdown event
    logger.info("Shutting down RAG Chatbot API...")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        description="A production-ready RAG chatbot backend API using LangChain & FAISS",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    # Setup CORS middleware
    raw_origins = getattr(settings, "ALLOWED_ORIGINS", "*")
    origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
    if not origins or "*" in origins:
        allow_origins = ["*"]
        allow_credentials = False
    else:
        allow_origins = origins
        allow_credentials = True

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
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

    # Security Headers Middleware
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("Content-Security-Policy", (
            "default-src 'self'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data:; "
            "frame-ancestors 'none'"
        ))
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"
        if settings.APP_ENV == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

    # Register endpoints router
    app.include_router(api_router, prefix="/api/v1")

    def docs_response(page: HTMLResponse) -> HTMLResponse:
        # Permit only this response's generated inline scripts. The main app keeps
        # its stricter self-only policy; Swagger/ReDoc also need their CDN assets.
        nonce = secrets.token_urlsafe(24)
        html = page.body.decode("utf-8").replace("<script", f'<script nonce="{nonce}"')
        return HTMLResponse(html, headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                "default-src 'self'; "
                f"script-src 'self' https://cdn.jsdelivr.net 'nonce-{nonce}'; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "img-src 'self' data:; worker-src 'self' blob:; "
                "object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
            ),
        })

    @app.get("/docs", include_in_schema=False)
    async def swagger_docs(request: Request):
        root_path = request.scope.get("root_path", "").rstrip("/")
        return docs_response(get_swagger_ui_html(
            openapi_url=f"{root_path}{app.openapi_url}", title=f"{app.title} - Swagger UI",
            swagger_favicon_url="data:,",
            oauth2_redirect_url=f"{root_path}/docs/oauth2-redirect",
        ))

    @app.get("/docs/oauth2-redirect", include_in_schema=False)
    async def swagger_redirect():
        return docs_response(get_swagger_ui_oauth2_redirect_html())

    @app.get("/redoc", include_in_schema=False)
    async def redoc_docs(request: Request):
        root_path = request.scope.get("root_path", "").rstrip("/")
        return docs_response(get_redoc_html(
            openapi_url=f"{root_path}{app.openapi_url}", title=f"{app.title} - ReDoc",
            redoc_favicon_url="data:,", with_google_fonts=False,
        ))

    # Serve the Atlas frontend from the static directory
    static_dir = Path(__file__).resolve().parent / "static"
    os.makedirs(static_dir, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=FileResponse, tags=["UI"])
    async def serve_index():
        index_file = static_dir / "index.html"
        return FileResponse(index_file)

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
        manifest_file = _os.path.join(settings.FAISS_INDEX_PATH, "manifest.json")
        if _os.path.exists(manifest_file) or _os.path.exists(index_file):
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
