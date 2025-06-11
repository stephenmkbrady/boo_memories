"""
Matrix Chat Database API - Simplified Version
A secure API service for storing and retrieving chat messages and media files
with automatic size management and cleanup.

SIMPLIFIED VERSION: Removed membership verification and token management features
"""

import os
import asyncio
import hashlib
import mimetypes
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path
import secrets
import shutil

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")
    print("⚠️ Falling back to system environment variables")

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Query, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, BigInteger, func, desc, asc, select, update
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel, Field
import aiofiles
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./chat_database.db")
MEDIA_DIRECTORY = os.getenv("MEDIA_DIRECTORY", "./media_files")
MAX_DATABASE_SIZE_MB = int(os.getenv("MAX_DATABASE_SIZE_MB", "1000"))  # 1GB default
API_KEY = os.getenv("API_KEY") or secrets.token_urlsafe(32)
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))

# Ensure media directory exists
Path(MEDIA_DIRECTORY).mkdir(parents=True, exist_ok=True)

# Database setup
Base = declarative_base()
engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Security
security = HTTPBearer()

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    room_id = Column(String(255), nullable=False, index=True)
    event_id = Column(String(255), nullable=False, unique=True, index=True)
    sender = Column(String(255), nullable=False, index=True)
    message_type = Column(String(50), nullable=False, index=True)  # text, image, file, audio, video, etc.
    content = Column(Text, nullable=True)
    media_filename = Column(String(500), nullable=True)
    media_mimetype = Column(String(100), nullable=True)
    media_size_bytes = Column(BigInteger, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseStats(Base):
    __tablename__ = "database_stats"
    
    id = Column(Integer, primary_key=True)
    total_messages = Column(Integer, default=0)
    total_media_files = Column(Integer, default=0)
    total_size_bytes = Column(BigInteger, default=0)
    last_cleanup = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow)

# Pydantic models
class MessageCreate(BaseModel):
    room_id: str
    event_id: str
    sender: str
    message_type: str
    content: Optional[str] = None
    timestamp: Optional[datetime] = None

class MessageResponse(BaseModel):
    id: int
    room_id: str
    event_id: str
    sender: str
    message_type: str
    content: Optional[str]
    media_filename: Optional[str]
    media_mimetype: Optional[str]
    media_size_bytes: Optional[int]
    timestamp: datetime
    created_at: datetime

class DatabaseStatsResponse(BaseModel):
    total_messages: int
    total_media_files: int
    total_size_mb: float
    last_cleanup: Optional[datetime]
    updated_at: datetime

class MediaUploadResponse(BaseModel):
    filename: str
    size: int
    mimetype: Optional[str]
    media_url: str
    message_id: int

# FastAPI app
app = FastAPI(
    title="Matrix Chat Database API",
    description="Simplified API for storing and retrieving Matrix chat messages and media",
    version="3.0.0"
)

# Add CORS middleware AFTER app is created
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://192.168.178.28:3000,http://localhost:3001,https://dash.example.com").split(",")

# For development, be more permissive
if os.getenv("ENVIRONMENT") == "development":
    # Allow common development origins
    ALLOWED_ORIGINS.extend([
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://127.0.0.1:3000",
        "http://192.168.178.28:3000",
        "http://192.168.178.28:3001",
        "https://dash.example.com"
    ])

print(f"🌐 CORS Allowed Origins: {ALLOWED_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Database dependency
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# Security dependency
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    logger.info(f"🔑 Received credentials: {credentials.credentials[:10]}..." if credentials else "No credentials")
    logger.info(f"🔑 Expected API key: {API_KEY[:10]}...")

    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

# Database operations
async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@app.on_event("startup")
async def startup_event():
    await create_tables()
    print(f"🔑 API Key: {API_KEY}")
    print(f"📁 Media Directory: {MEDIA_DIRECTORY}")
    print(f"🗄️ Database URL: {DATABASE_URL}")
    print("✅ Simplified Database API started successfully!")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.utcnow(), 
        "features": ["messages", "media", "simplified_api"],
        "version": "3.0.0"
    }

# Simplified messages endpoint - no membership verification
@app.get("/messages")
async def get_messages(
    room_id: str = Query(..., description="Room ID (required)"),
    limit: int = Query(100, description="Number of messages to return"),
    include_media: bool = Query(False, description="Include media file information"),
    message_type: str = Query(None, description="Filter by message type (text, image, file, audio, video)"),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(verify_api_key)
):
    """Get messages for a specific room - simplified version without membership verification"""
    from sqlalchemy import select, desc
    
    try:
        if not room_id:
            raise HTTPException(
                status_code=400, 
                detail="room_id is required"
            )
        
        # Build query - filter by room_id
        query = select(Message).where(Message.room_id == room_id).order_by(desc(Message.timestamp)).limit(limit)
            
        if message_type:
            query = query.where(Message.message_type == message_type)
        
        result = await db.execute(query)
        messages = result.scalars().all()
        
        # Convert to dict format
        message_list = []
        for msg in messages:
            message_dict = {
                "id": msg.id,
                "room_id": msg.room_id,
                "event_id": msg.event_id,
                "sender": msg.sender,
                "message_type": msg.message_type,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "created_at": msg.created_at
            }
            
            # Include media information if requested or if message has media
            if include_media or msg.media_filename:
                message_dict.update({
                    "media_filename": msg.media_filename,
                    "media_mimetype": msg.media_mimetype,
                    "media_size_bytes": msg.media_size_bytes,
                    "media_url": f"/media/{msg.media_filename}" if msg.media_filename else None
                })
            
            message_list.append(message_dict)
        
        response_data = {
            "messages": message_list,
            "count": len(message_list),
            "room_id": room_id,
            "message_type": message_type,
            "include_media": include_media
        }
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting messages: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve messages: {str(e)}")

@app.get("/messages/{message_id}")
async def get_message(
    message_id: int,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(verify_api_key)
):
    """Get a specific message by ID"""
    try:
        query = select(Message).where(Message.id == message_id)
        result = await db.execute(query)
        message = result.scalar_one_or_none()
        
        if not message:
            raise HTTPException(status_code=404, detail=f"Message {message_id} not found")
        
        message_dict = {
            "id": message.id,
            "room_id": message.room_id,
            "event_id": message.event_id,
            "sender": message.sender,
            "message_type": message.message_type,
            "content": message.content,
            "media_filename": message.media_filename,
            "media_mimetype": message.media_mimetype,
            "media_size_bytes": message.media_size_bytes,
            "media_url": f"/media/{message.media_filename}" if message.media_filename else None,
            "timestamp": message.timestamp,
            "created_at": message.created_at
        }
        
        return message_dict
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting message {message_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve message: {str(e)}")

@app.get("/media/{filename}")
async def download_media(filename: str):
    """Download a media file"""
    try:
        file_path = Path(MEDIA_DIRECTORY) / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Security check: ensure the file is within the media directory
        if not str(file_path.resolve()).startswith(str(Path(MEDIA_DIRECTORY).resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
        
        return FileResponse(
            file_path, 
            filename=filename,
            media_type=mimetypes.guess_type(filename)[0] or 'application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error downloading media {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download file: {str(e)}")

@app.post("/media/upload", response_model=MediaUploadResponse)  
async def upload_media(
    message_id: int = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(verify_api_key)
):
    """Upload a media file and link it to a message"""
    try:
        # Check if message exists
        query = select(Message).where(Message.id == message_id)
        result = await db.execute(query)
        message = result.scalar_one_or_none()
        
        if not message:
            raise HTTPException(status_code=404, detail=f"Message {message_id} not found")
        
        # Validate file size
        content = await file.read()
        file_size = len(content)
        max_size = MAX_FILE_SIZE_MB * 1024 * 1024
        
        if file_size > max_size:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB"
            )
        
        # Generate a unique filename
        file_extension = Path(file.filename).suffix if file.filename else ""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{message_id}_{timestamp}_{secrets.token_urlsafe(8)}{file_extension}"
        file_path = Path(MEDIA_DIRECTORY) / filename
        
        # Save the file
        async with aiofiles.open(file_path, "wb") as buffer:
            await buffer.write(content)
        
        # Update the message record with media information
        update_stmt = update(Message).where(Message.id == message_id).values(
            media_filename=filename,
            media_mimetype=file.content_type,
            media_size_bytes=file_size
        )
        await db.execute(update_stmt)
        await db.commit()
        
        logger.info(f"📎 Media uploaded and linked to message {message_id}: {filename} ({file_size} bytes)")
        
        return MediaUploadResponse(
            filename=filename, 
            size=file_size,
            mimetype=file.content_type,
            media_url=f"/media/{filename}",
            message_id=message_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error uploading media: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/messages", response_model=MessageResponse)
async def create_message(
    message: MessageCreate,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(verify_api_key)
):
    """Store a new chat message"""
    try:
        # Check for duplicate event_id
        existing_query = select(Message).where(Message.event_id == message.event_id)
        existing_result = await db.execute(existing_query)
        existing_message = existing_result.scalar_one_or_none()
        
        if existing_message:
            logger.info(f"⚠️ Duplicate event_id {message.event_id}, returning existing message")
            return existing_message
        
        db_message = Message(
            room_id=message.room_id,
            event_id=message.event_id,
            sender=message.sender,
            message_type=message.message_type,
            content=message.content,
            timestamp=message.timestamp or datetime.utcnow()
        )
        
        db.add(db_message)
        await db.commit()
        await db.refresh(db_message)
        
        logger.info(f"💾 Stored message: {message.message_type} from {message.sender} (ID: {db_message.id})")
        
        return db_message
        
    except Exception as e:
        logger.error(f"❌ Error creating message: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create message: {str(e)}")

@app.get("/stats", response_model=DatabaseStatsResponse)
async def get_database_stats(
    db: AsyncSession = Depends(get_db),
    _: None = Depends(verify_api_key)
):
    """Get database statistics and size information"""
    try:
        # Count total messages
        message_count_query = select(func.count(Message.id))
        message_count_result = await db.execute(message_count_query)
        message_count = message_count_result.scalar() or 0
        
        # Count media files (messages with media_filename not null)
        media_count_query = select(func.count(Message.id)).where(Message.media_filename.isnot(None))
        media_count_result = await db.execute(media_count_query)
        media_count = media_count_result.scalar() or 0
        
        # Calculate total database size
        total_size = 0
        
        # Database file size
        if DATABASE_URL.startswith("sqlite"):
            # For SQLite, get file size
            db_file = DATABASE_URL.replace("sqlite+aiosqlite:///", "").replace("./", "")
            if os.path.exists(db_file):
                total_size += os.path.getsize(db_file)
        else:
            # For PostgreSQL, try to get database size
            try:
                from sqlalchemy import text
                size_result = await db.execute(text("SELECT pg_database_size(current_database())"))
                db_size = size_result.scalar()
                if db_size:
                    total_size += db_size
            except Exception as e:
                logger.warning(f"⚠️ Could not get PostgreSQL database size: {e}")
        
        # Add media files size
        media_dir = Path(MEDIA_DIRECTORY)
        media_total_size = 0
        media_file_count = 0
        
        if media_dir.exists():
            for file_path in media_dir.glob("*"):
                if file_path.is_file():
                    try:
                        file_size = file_path.stat().st_size
                        media_total_size += file_size
                        media_file_count += 1
                    except:
                        pass
        
        total_size += media_total_size
        
        logger.info(f"📊 Stats: {message_count} messages, {media_count} with media, {media_file_count} files, {total_size/1024/1024:.2f}MB total")
        
        return DatabaseStatsResponse(
            total_messages=message_count,
            total_media_files=media_count,
            total_size_mb=round(total_size / (1024 * 1024), 2),
            last_cleanup=None,
            updated_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting database stats: {e}")
        # Return basic stats even if detailed calculation fails
        return DatabaseStatsResponse(
            total_messages=0,
            total_media_files=0,
            total_size_mb=0.0,
            last_cleanup=None,
            updated_at=datetime.utcnow()
        )

@app.delete("/messages/{message_id}")
async def delete_message(
    message_id: int,
    delete_media: bool = Query(False, description="Also delete associated media file"),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(verify_api_key)
):
    """Delete a message and optionally its media file"""
    try:
        # Get the message first
        query = select(Message).where(Message.id == message_id)
        result = await db.execute(query)
        message = result.scalar_one_or_none()
        
        if not message:
            raise HTTPException(status_code=404, detail=f"Message {message_id} not found")
        
        # Delete media file if requested and exists
        if delete_media and message.media_filename:
            media_path = Path(MEDIA_DIRECTORY) / message.media_filename
            if media_path.exists():
                try:
                    media_path.unlink()
                    logger.info(f"🗑️ Deleted media file: {message.media_filename}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not delete media file {message.media_filename}: {e}")
        
        # Delete the message record
        await db.delete(message)
        await db.commit()
        
        logger.info(f"🗑️ Deleted message {message_id}")
        
        return {"message": f"Message {message_id} deleted", "media_deleted": delete_media}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting message {message_id}: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete message: {str(e)}")

@app.post("/cleanup")
async def trigger_cleanup(
    max_age_days: int = Query(30, description="Delete messages older than this many days"),
    max_size_mb: int = Query(None, description="Target maximum total size in MB"),
    dry_run: bool = Query(False, description="Show what would be deleted without actually deleting"),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(verify_api_key)
):
    """Trigger database and media cleanup"""
    try:
        from sqlalchemy import text
        
        cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
        
        # Find old messages
        old_messages_query = select(Message).where(Message.timestamp < cutoff_date)
        result = await db.execute(old_messages_query)
        old_messages = result.scalars().all()
        
        cleanup_stats = {
            "messages_to_delete": len(old_messages),
            "media_files_to_delete": 0,
            "estimated_space_freed_mb": 0,
            "dry_run": dry_run
        }
        
        if not dry_run and old_messages:
            # Delete media files and count space
            for message in old_messages:
                if message.media_filename:
                    media_path = Path(MEDIA_DIRECTORY) / message.media_filename
                    if media_path.exists():
                        try:
                            file_size = media_path.stat().st_size
                            cleanup_stats["estimated_space_freed_mb"] += file_size / (1024 * 1024)
                            cleanup_stats["media_files_to_delete"] += 1
                            
                            if not dry_run:
                                media_path.unlink()
                        except Exception as e:
                            logger.warning(f"⚠️ Could not delete media file {message.media_filename}: {e}")
            
            # Delete message records
            if not dry_run:
                for message in old_messages:
                    await db.delete(message)
                await db.commit()
        
        logger.info(f"🧹 Cleanup {'simulation' if dry_run else 'completed'}: {cleanup_stats}")
        
        return cleanup_stats
        
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")
        if not dry_run:
            await db.rollback()
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Simplified Matrix Chat Database API")
    uvicorn.run(app, host="0.0.0.0", port=8000)
