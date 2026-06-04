# Bu dosya: FastAPI ana uygulama giriş noktası ve endpoint tanımları

# arif live demo comment 1

import os
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.metrics import RESIZE_DURATION, RESIZE_TOTAL, S3_UPLOAD_DURATION, setup_metrics
from app.models import ImageRecord
from app.schemas import ImageListResponse, ImageResponse
from app.services import ImageResizeService, S3Service

S3_BUCKET = os.getenv("S3_BUCKET", "image-resize-bucket")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB maximum size limit to protect server memory (DoS prevention)

s3_service = S3Service()
resize_service = ImageResizeService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager that handles startup database migrations and S3 bucket readiness.

    Args:
        app: The FastAPI application instance.
    """
    # Create DB tables
    Base.metadata.create_all(bind=engine)
    # Ensure S3 bucket exists
    try:
        s3_service.ensure_bucket(S3_BUCKET)
    except Exception as e:
        # Log error during startup
        print(f"Lifespan startup error ensuring S3 bucket {S3_BUCKET}: {e}")
    yield

app = FastAPI(
    title="Image Resize Service",
    description="Microservice for image resizing with S3 storage and DB tracking.",
    lifespan=lifespan
)

# CORS config with strict origins instead of wildcards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize HTTP request instrumentation and register custom metrics
setup_metrics(app)

@app.get("/", response_class=HTMLResponse)
def get_ui():
    """Serves the static Web UI file for user interactions.

    Returns:
        HTMLResponse: Web interface.
    """
    ui_path = os.path.join(os.path.dirname(__file__), "..", "ui", "index.html")
    if not os.path.exists(ui_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Static Web UI index.html not found."
        )
    with open(ui_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/resize", response_model=ImageResponse, status_code=status.HTTP_200_OK)
async def resize_image(
    file: UploadFile = File(...),
    width: int = Form(...),
    height: int = Form(...),
    quality: int = Form(85),
    format: str = Form("JPEG"),
    db: Session = Depends(get_db)
):
    """Processes, resizes and uploads an image. Saves metadata to database.

    Returns:
        ImageResponse: Formatted metadata including download URL.
    """
    contents = await file.read()
    file_size = len(contents)

    # Protect against massive files (DoS prevention)
    if file_size > MAX_FILE_SIZE:
        RESIZE_TOTAL.labels(status="failed", format=format).inc()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File size exceeds the maximum limit of 10MB."
        )

    clean_format = format.upper()
    if clean_format not in ("JPEG", "PNG", "WEBP"):
        RESIZE_TOTAL.labels(status="failed", format=format).inc()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format '{format}'. Allowed formats: JPEG, PNG, WEBP."
        )

    # Validate parameters
    if not (0 < width <= 4096) or not (0 < height <= 4096):
        RESIZE_TOTAL.labels(status="failed", format=format).inc()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Dimensions must be between 1 and 4096."
        )

    if not (1 <= quality <= 100):
        RESIZE_TOTAL.labels(status="failed", format=format).inc()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Quality must be between 1 and 100."
        )

    # Extract original dimensions to validate file content is a valid image
    try:
        orig_w, orig_h = resize_service.get_image_dimensions(contents)
    except ValueError:
        RESIZE_TOTAL.labels(status="failed", format=format).inc()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid image."
        )

    # Perform resizing and record performance metric
    start_time = time.perf_counter()
    try:
        resized_data, final_w, final_h = resize_service.resize(
            contents, width, height, quality, clean_format
        )
    except ValueError as e:
        RESIZE_TOTAL.labels(status="failed", format=format).inc()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    duration = time.perf_counter() - start_time
    RESIZE_DURATION.observe(duration)

    # Generate unique, unpredictable filename to prevent path injection / key clashes
    original_filename = os.path.basename(file.filename or "image")
    unique_key = f"resized/{uuid.uuid4().hex}_{original_filename}"
    content_type = f"image/{clean_format.lower()}"

    # Upload to S3 storage and record upload latency
    s3_start = time.perf_counter()
    try:
        s3_service.upload(S3_BUCKET, unique_key, resized_data, content_type)
    except Exception:
        RESIZE_TOTAL.labels(status="failed", format=format).inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store resized image."
        )
    s3_duration = time.perf_counter() - s3_start
    S3_UPLOAD_DURATION.observe(s3_duration)

    # Save details to DB
    try:
        record = ImageRecord(
            original_filename=original_filename,
            s3_bucket=S3_BUCKET,
            s3_key=unique_key,
            original_width=orig_w,
            original_height=orig_h,
            resized_width=final_w,
            resized_height=final_h,
            file_size_bytes=len(resized_data),
            content_type=content_type
        )
        db.add(record)
        db.commit()
        db.refresh(record)
    except Exception:
        traceback.print_exc()
        # Rollback S3 upload if metadata database save fails
        try:
            s3_service.delete(S3_BUCKET, unique_key)
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save image record metadata."
        )

    # Generate temporary pre-signed GET URL for client consumption
    try:
        presigned_url = s3_service.generate_presigned_url(S3_BUCKET, unique_key)
    except Exception:
        presigned_url = None

    RESIZE_TOTAL.labels(status="success", format=format).inc()

    response_data = ImageResponse.model_validate(record)
    response_data.presigned_url = presigned_url
    return response_data

@app.get("/images", response_model=ImageListResponse)
def get_images(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Returns a list of image metadata records.

    Args:
        skip: Pagination offset.
        limit: Number of records to return.
        db: Database session.
    """
    try:
        total = db.query(ImageRecord).count()
        records = db.query(ImageRecord).offset(skip).limit(limit).all()
    except Exception:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to query image records."
        )

    items = []
    for record in records:
        try:
            presigned_url = s3_service.generate_presigned_url(record.s3_bucket, record.s3_key)
        except Exception:
            presigned_url = None
        
        item = ImageResponse.model_validate(record)
        item.presigned_url = presigned_url
        items.append(item)

    return ImageListResponse(items=items, total=total)

@app.get("/images/{image_id}", response_model=ImageResponse)
def get_image(image_id: int, db: Session = Depends(get_db)):
    """Fetches details of a single image metadata record.

    Args:
        image_id: DB ID of the image record.
        db: Database session.
    """
    try:
        record = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
    except Exception:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error reading record from database."
        )

    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image record with ID {image_id} not found."
        )

    try:
        presigned_url = s3_service.generate_presigned_url(record.s3_bucket, record.s3_key)
    except Exception:
        presigned_url = None

    response_data = ImageResponse.model_validate(record)
    response_data.presigned_url = presigned_url
    return response_data

@app.delete("/images/{image_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_image(image_id: int, db: Session = Depends(get_db)):
    """Removes the image record from database and its file from storage.

    Args:
        image_id: DB ID of the image record to delete.
        db: Database session.
    """
    try:
        record = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
    except Exception:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error accessing database."
        )

    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image record with ID {image_id} not found."
        )

    # Delete from S3 storage
    try:
        s3_service.delete(record.s3_bucket, record.s3_key)
    except Exception:
        pass

    # Delete from DB
    try:
        db.delete(record)
        db.commit()
    except Exception:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete image record metadata."
        )

    return Response(status_code=status.HTTP_204_NO_CONTENT)

@app.get("/health")
def health_check():
    """Smoke test check endpoint to query microservice status.

    Returns:
        dict: Status indicators and current UTC time.
    """
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
