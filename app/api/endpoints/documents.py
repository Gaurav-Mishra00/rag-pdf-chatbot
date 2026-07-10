from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from app.api.deps import get_pdf_processor, get_vector_store
from app.core.security import verify_api_key
from app.schemas.document import DocumentUploadResponse, IngestionStatus
from app.services.pdf_processor import PDFProcessorService
from app.vectorstore.faiss_store import FAISSVectorStore

router = APIRouter()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_api_key)],
)
async def upload_document(
    file: UploadFile = File(...),
    pdf_processor: PDFProcessorService = Depends(get_pdf_processor),
    vector_store: FAISSVectorStore = Depends(get_vector_store),
) -> DocumentUploadResponse:
    """
    Uploads a PDF file, parses/chunks its text content, generates embeddings,
    and updates the FAISS vector index.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    # Read file bytes
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read file: {str(e)}",
        )

    # Process and chunk PDF
    documents = pdf_processor.process_pdf(content, filename=file.filename)

    # Add to FAISS index
    if documents:
        vector_store.add_documents(documents)
        status_info = IngestionStatus.COMPLETED
        message = f"Successfully parsed and indexed {len(documents)} chunks."
    else:
        status_info = IngestionStatus.FAILED
        message = "Failed to parse text from the PDF file or document was empty."

    return DocumentUploadResponse(
        filename=file.filename,
        status=status_info,
        message=message,
        document_id="mock-uuid-for-doc",
    )
