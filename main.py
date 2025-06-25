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
import yaml
import httpx
from jose import JWTError, jwt

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

# Load configuration
def load_config():
    config_file = os.getenv("CONFIG_FILE", "config.yaml")
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"✅ Loaded configuration from {config_file}")
        return config
    except FileNotFoundError:
        logger.warning(f"⚠️ Config file {config_file} not found, using defaults")
        return {}
    except Exception as e:
        logger.error(f"❌ Error loading config: {e}")
        return {}

config = load_config()

# Configuration with YAML overrides
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./chat_database.db")
MEDIA_DIRECTORY = os.getenv("MEDIA_DIRECTORY", config.get("media", {}).get("directory", "./media_files"))
MAX_DATABASE_SIZE_MB = int(os.getenv("MAX_DATABASE_SIZE_MB", config.get("database", {}).get("max_size_mb", 1000)))
API_KEY = os.getenv("API_KEY") or secrets.token_urlsafe(32)
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", config.get("media", {}).get("max_file_size_mb", 50)))

# PIN Configuration
PIN_CONFIG = config.get("pin_auth", {})
PIN_ENABLED = PIN_CONFIG.get("enabled", True)
PIN_EXPIRY_HOURS = PIN_CONFIG.get("expiry_hours", 24)
PIN_CLEANUP_INTERVAL_HOURS = PIN_CONFIG.get("cleanup_interval_hours", 1)
PIN_RATE_LIMIT_PER_HOUR = PIN_CONFIG.get("rate_limit_per_hour", 3)

# NIST Beacon Configuration
NIST_CONFIG = config.get("nist_beacon", {})
NIST_BEACON_URL = NIST_CONFIG.get("url", "https://beacon.nist.gov/beacon/2.0/chain/last/pulse")
NIST_BEACON_TIMEOUT = NIST_CONFIG.get("timeout_seconds", 10)

# JWT Configuration
JWT_CONFIG = config.get("jwt", {})
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY") or JWT_CONFIG.get("secret_key") or secrets.token_urlsafe(32)
JWT_ALGORITHM = JWT_CONFIG.get("algorithm", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_HOURS = JWT_CONFIG.get("access_token_expire_hours", 24)

# Ensure media directory exists
Path(MEDIA_DIRECTORY).mkdir(parents=True, exist_ok=True)

# Database setup
Base = declarative_base()
engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Database Models (defined early to avoid forward reference issues)
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

class RoomPin(Base):
    __tablename__ = "room_pins"
    
    room_id = Column(String(255), primary_key=True, index=True)
    pin_code = Column(String(6), nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)
    request_count = Column(Integer, default=1)
    last_request_at = Column(DateTime, default=datetime.utcnow)
    nist_beacon_value = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseStats(Base):
    __tablename__ = "database_stats"
    
    id = Column(Integer, primary_key=True)
    total_messages = Column(Integer, default=0)
    total_media_files = Column(Integer, default=0)
    total_size_bytes = Column(BigInteger, default=0)
    last_cleanup = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow)

# Security
security = HTTPBearer()

# PIN Management Functions
async def fetch_nist_beacon_value() -> Optional[str]:
    """Fetch random value from NIST beacon API for PIN generation seed"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(NIST_BEACON_URL, timeout=NIST_BEACON_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            return data.get("pulse", {}).get("outputValue")
    except Exception as e:
        logger.warning(f"⚠️ NIST beacon failed: {e}, using fallback")
        return None

async def generate_room_pin(room_id: str, nist_value: Optional[str] = None) -> str:
    """Generate 6-digit PIN using NIST beacon + room_id for entropy"""
    if not nist_value:
        nist_value = await fetch_nist_beacon_value()
    
    if nist_value:
        # Use NIST beacon value + room_id for deterministic but secure PIN
        seed_string = f"{nist_value}_{room_id}_{datetime.utcnow().strftime('%Y-%m-%d')}"
        hash_object = hashlib.sha256(seed_string.encode())
        hex_dig = hash_object.hexdigest()
        pin_num = int(hex_dig[:8], 16) % 1000000
    else:
        # Fallback to Python secrets
        pin_num = secrets.randbelow(1000000)
    
    return str(pin_num).zfill(6)

async def check_pin_rate_limit(room_id: str, db: AsyncSession) -> bool:
    """Check if room has exceeded PIN request rate limit"""
    try:
        from sqlalchemy import select
        
        # Get current room pin record
        query = select(RoomPin).where(RoomPin.room_id == room_id)
        result = await db.execute(query)
        room_pin = result.scalar_one_or_none()
        
        if not room_pin:
            return True  # No previous requests, allow
        
        # Check if within rate limit window
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        if room_pin.last_request_at < one_hour_ago:
            # Reset counter if outside window
            return True
        
        # Check rate limit
        return room_pin.request_count < PIN_RATE_LIMIT_PER_HOUR
        
    except Exception as e:
        logger.error(f"❌ Error checking rate limit: {e}")
        return False

async def get_or_create_room_pin(room_id: str, db: AsyncSession) -> RoomPin:
    """Get current valid PIN for room or create new one"""
    try:
        from sqlalchemy import select, update
        
        # Check for existing valid PIN
        query = select(RoomPin).where(
            RoomPin.room_id == room_id,
            RoomPin.expires_at > datetime.utcnow()
        )
        result = await db.execute(query)
        existing_pin = result.scalar_one_or_none()
        
        if existing_pin:
            # Update request count and timestamp
            update_stmt = update(RoomPin).where(RoomPin.room_id == room_id).values(
                request_count=RoomPin.request_count + 1,
                last_request_at=datetime.utcnow()
            )
            await db.execute(update_stmt)
            await db.commit()
            return existing_pin
        
        # Create new PIN
        nist_value = await fetch_nist_beacon_value()
        pin_code = await generate_room_pin(room_id, nist_value)
        expires_at = datetime.utcnow() + timedelta(hours=PIN_EXPIRY_HOURS)
        
        # Delete any existing expired PIN for this room
        from sqlalchemy import delete
        delete_stmt = delete(RoomPin).where(RoomPin.room_id == room_id)
        await db.execute(delete_stmt)
        
        # Create new PIN record
        new_pin = RoomPin(
            room_id=room_id,
            pin_code=pin_code,
            expires_at=expires_at,
            request_count=1,
            last_request_at=datetime.utcnow(),
            nist_beacon_value=nist_value,
            created_at=datetime.utcnow()
        )
        
        db.add(new_pin)
        await db.commit()
        await db.refresh(new_pin)
        
        logger.info(f"📌 Generated new PIN for room {room_id[:20]}... (expires: {expires_at})")
        return new_pin
        
    except Exception as e:
        logger.error(f"❌ Error managing room PIN: {e}")
        await db.rollback()
        raise

async def validate_pin(room_id: str, pin: str, db: AsyncSession) -> bool:
    """Validate PIN against current room PIN"""
    try:
        from sqlalchemy import select
        
        query = select(RoomPin).where(
            RoomPin.room_id == room_id,
            RoomPin.pin_code == pin,
            RoomPin.expires_at > datetime.utcnow()
        )
        result = await db.execute(query)
        valid_pin = result.scalar_one_or_none()
        
        return valid_pin is not None
        
    except Exception as e:
        logger.error(f"❌ Error validating PIN: {e}")
        return False

async def create_room_access_token(room_id: str) -> str:
    """Create JWT token with room_id claim"""
    expires_at = datetime.utcnow() + timedelta(hours=JWT_ACCESS_TOKEN_EXPIRE_HOURS)
    payload = {
        "room_id": room_id,
        "exp": expires_at,
        "iat": datetime.utcnow(),
        "type": "room_access"
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

async def validate_room_access_token(token: str) -> Optional[str]:
    """Validate JWT token and return room_id"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        if payload.get("type") != "room_access":
            return None
        return payload.get("room_id")
    except JWTError as e:
        # Only log failed tokens with their endings to track different token sources
        token_ending = token[-10:] if len(token) > 10 else token
        logger.warning(f"⚠️ Invalid JWT token ending in: {token_ending} - Error: {e}")
        return None

async def cleanup_expired_pins(db: AsyncSession) -> int:
    """Remove expired PINs from database"""
    try:
        from sqlalchemy import delete
        
        delete_stmt = delete(RoomPin).where(RoomPin.expires_at <= datetime.utcnow())
        result = await db.execute(delete_stmt)
        await db.commit()
        
        deleted_count = result.rowcount
        if deleted_count > 0:
            logger.info(f"🧹 Cleaned up {deleted_count} expired PINs")
        
        return deleted_count
        
    except Exception as e:
        logger.error(f"❌ Error during PIN cleanup: {e}")
        await db.rollback()
        return 0

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

class PinRequest(BaseModel):
    room_id: str
    pin: str

class PinResponse(BaseModel):
    pin: str
    expires_at: datetime
    room_id: str

class TokenResponse(BaseModel):
    access_token: str
    expires_at: datetime
    room_id: str

# FastAPI app
app = FastAPI(
    title="Matrix Chat Database API",
    description="Simplified API for storing and retrieving Matrix chat messages and media",
    version="3.0.0"
)

# CORS Configuration - Environment variable override or config.yaml fallback
CORS_CONFIG = config.get("cors", {})
DEFAULT_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001", 
    "http://127.0.0.1:3000",
    "https://dash.example.com"
]

# Use environment variable if set, otherwise use config.yaml, otherwise use defaults
if os.getenv("ALLOWED_ORIGINS"):
    ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS").split(",")]
    logger.info(f"🌐 Using CORS origins from environment variable")
else:
    config_origins = CORS_CONFIG.get("allowed_origins", [])
    ALLOWED_ORIGINS = config_origins if config_origins else DEFAULT_ORIGINS
    logger.info(f"🌐 Using CORS origins from config.yaml")

# Development mode adds extra permissive origins
if CORS_CONFIG.get("development_mode", True) or os.getenv("ENVIRONMENT") == "development":
    development_origins = [
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://127.0.0.1:3000",
        "http://192.168.178.28:3000",
        "http://192.168.178.28:3001"
    ]
    for origin in development_origins:
        if origin not in ALLOWED_ORIGINS:
            ALLOWED_ORIGINS.append(origin)

# Remove duplicates while preserving order
ALLOWED_ORIGINS = list(dict.fromkeys(ALLOWED_ORIGINS))
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

# Background PIN cleanup task
async def periodic_pin_cleanup():
    """Background task to cleanup expired PINs"""
    while True:
        try:
            await asyncio.sleep(PIN_CLEANUP_INTERVAL_HOURS * 3600)  # Convert hours to seconds
            async with AsyncSessionLocal() as db:
                deleted_count = await cleanup_expired_pins(db)
                if deleted_count > 0:
                    logger.info(f"🧹 Background cleanup removed {deleted_count} expired PINs")
        except Exception as e:
            logger.error(f"❌ Background PIN cleanup error: {e}")
            await asyncio.sleep(300)  # Wait 5 minutes before retrying

@app.on_event("startup")
async def startup_event():
    await create_tables()
    
    # Start background PIN cleanup task if PIN auth is enabled
    if PIN_ENABLED:
        asyncio.create_task(periodic_pin_cleanup())
        logger.info(f"🧹 Started background PIN cleanup (interval: {PIN_CLEANUP_INTERVAL_HOURS}h)")
    
    print(f"🔑 API Key: {API_KEY}")
    print(f"📁 Media Directory: {MEDIA_DIRECTORY}")
    print(f"🗄️ Database URL: {DATABASE_URL}")
    print(f"📌 PIN Authentication: {'Enabled' if PIN_ENABLED else 'Disabled'}")
    print(f"🔐 JWT Secret: {JWT_SECRET_KEY[:10]}...")
    print("✅ Database API with PIN Authentication started successfully!")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.utcnow(), 
        "features": ["messages", "media", "simplified_api", "pin_auth"],
        "version": "3.0.0",
        "pin_auth_enabled": PIN_ENABLED
    }

# PIN Authentication Endpoints
@app.post("/rooms/{room_id}/pin", response_model=PinResponse)
async def request_room_pin(
    room_id: str,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(verify_api_key)
):
    """Generate/retrieve current PIN for room (bot-only endpoint)"""
    if not PIN_ENABLED:
        raise HTTPException(status_code=503, detail="PIN authentication is disabled")
    
    try:
        # Check rate limit
        if not await check_pin_rate_limit(room_id, db):
            raise HTTPException(
                status_code=429, 
                detail="Rate limit exceeded. Maximum 3 PIN requests per hour per room."
            )
        
        # Get or create PIN
        room_pin = await get_or_create_room_pin(room_id, db)
        
        logger.info(f"📌 PIN requested for room {room_id[:20]}... (requests: {room_pin.request_count})")
        
        return PinResponse(
            pin=room_pin.pin_code,
            expires_at=room_pin.expires_at,
            room_id=room_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error requesting PIN for room {room_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate PIN")

@app.post("/internal/auth/pin", response_model=TokenResponse)
async def validate_pin_and_get_token(
    pin_request: PinRequest,
    db: AsyncSession = Depends(get_db)
):
    """Internal endpoint: Validate PIN for web UI access"""
    if not PIN_ENABLED:
        raise HTTPException(status_code=503, detail="PIN authentication is disabled")
    
    try:
        # Validate PIN
        is_valid = await validate_pin(pin_request.room_id, pin_request.pin, db)
        
        if not is_valid:
            raise HTTPException(
                status_code=401, 
                detail="Invalid PIN or PIN expired"
            )
        
        # Create room access token
        access_token = await create_room_access_token(pin_request.room_id)
        expires_at = datetime.utcnow() + timedelta(hours=JWT_ACCESS_TOKEN_EXPIRE_HOURS)
        
        logger.info(f"🔑 Room access token issued for {pin_request.room_id[:20]}...")
        
        return TokenResponse(
            access_token=access_token,
            expires_at=expires_at,
            room_id=pin_request.room_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error validating PIN: {e}")
        raise HTTPException(status_code=500, detail="Failed to validate PIN")

# Token validation dependency for UI endpoints
async def verify_room_access_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify room access token and return room_id"""
    if not credentials:
        raise HTTPException(status_code=401, detail="No authorization token provided")
    
    room_id = await validate_room_access_token(credentials.credentials)
    if not room_id:
        raise HTTPException(status_code=401, detail="Invalid or expired access token")
    
    return room_id

# Web UI Proxy Endpoints (using room access tokens)
@app.get("/ui/rooms/{room_id}/messages")
async def get_room_messages_ui(
    room_id: str,
    limit: int = Query(100, description="Number of messages to return"),
    include_media: bool = Query(False, description="Include media file information"),
    message_type: str = Query(None, description="Filter by message type"),
    token_room_id: str = Depends(verify_room_access_token),
    db: AsyncSession = Depends(get_db)
):
    """Get messages for specific room via UI token authentication"""
    if token_room_id != room_id:
        raise HTTPException(status_code=403, detail="Token not valid for this room")
    
    # Reuse existing message logic
    try:
        from sqlalchemy import select, desc
        
        query = select(Message).where(Message.room_id == room_id).order_by(desc(Message.timestamp)).limit(limit)
            
        if message_type:
            query = query.where(Message.message_type == message_type)
        
        result = await db.execute(query)
        messages = result.scalars().all()
        
        # Convert to dict format (same as existing endpoint)
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
            
            if include_media or msg.media_filename:
                message_dict.update({
                    "media_filename": msg.media_filename,
                    "media_mimetype": msg.media_mimetype,
                    "media_size_bytes": msg.media_size_bytes,
                    "media_url": f"/media/{msg.media_filename}" if msg.media_filename else None
                })
            
            message_list.append(message_dict)
        
        return {
            "messages": message_list,
            "count": len(message_list),
            "room_id": room_id,
            "message_type": message_type,
            "include_media": include_media,
            "auth_method": "room_token"
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting UI messages for room {room_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve messages")

@app.get("/ui/rooms/{room_id}/stats")
async def get_room_stats_ui(
    room_id: str,
    token_room_id: str = Depends(verify_room_access_token),
    db: AsyncSession = Depends(get_db)
):
    """Get database stats for specific room via UI token authentication"""
    if token_room_id != room_id:
        raise HTTPException(status_code=403, detail="Token not valid for this room")
    
    # Return the same stats as the main endpoint but with room context
    try:
        # Count total messages
        message_count_query = select(func.count(Message.id))
        message_count_result = await db.execute(message_count_query)
        message_count = message_count_result.scalar() or 0
        
        # Count media files (messages with media_filename not null)
        media_count_query = select(func.count(Message.id)).where(Message.media_filename.isnot(None))
        media_count_result = await db.execute(media_count_query)
        media_count = media_count_result.scalar() or 0
        
        # Count room-specific messages
        room_message_count_query = select(func.count(Message.id)).where(Message.room_id == room_id)
        room_message_count_result = await db.execute(room_message_count_query)
        room_message_count = room_message_count_result.scalar() or 0
        
        # Get media directory size (reuse existing logic)
        media_size = 0
        if os.path.exists(MEDIA_DIRECTORY):
            for root, dirs, files in os.walk(MEDIA_DIRECTORY):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        media_size += os.path.getsize(file_path)
        
        # Get database file size (simplified)
        database_size = 0
        if "sqlite" in DATABASE_URL.lower():
            db_path = DATABASE_URL.replace("sqlite+aiosqlite:///", "").replace("./", "")
            if os.path.exists(db_path):
                database_size = os.path.getsize(db_path)
        
        return {
            "message_count": message_count,
            "room_message_count": room_message_count,
            "media_count": media_count,
            "database_size_mb": round(database_size / (1024 * 1024), 2),
            "media_size_mb": round(media_size / (1024 * 1024), 2),
            "room_id": room_id,
            "auth_method": "room_token"
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting UI stats for room {room_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve stats")

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

async def verify_media_access(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify access for media downloads - supports both API keys and JWT tokens"""
    if not credentials:
        raise HTTPException(status_code=401, detail="No authorization token provided")
    
    token = credentials.credentials
    
    # Try API key first (for backward compatibility)
    if token == API_KEY:
        logger.info(f"🔑 Media access authorized with API key")
        return "api_key"
    
    # Try JWT token validation
    room_id = await validate_room_access_token(token)
    if room_id:
        logger.info(f"🔑 Media access authorized with JWT token for room {room_id[:20]}...")
        return room_id
    
    # If neither worked, deny access
    raise HTTPException(
        status_code=401, 
        detail="Invalid authorization token - use valid API key or room access token"
    )

@app.get("/media/{filename}")
async def download_media(
    filename: str,
    auth_info: str = Depends(verify_media_access)
):
    """Download a media file with proper authentication"""
    try:
        file_path = Path(MEDIA_DIRECTORY) / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Security check: ensure the file is within the media directory
        if not str(file_path.resolve()).startswith(str(Path(MEDIA_DIRECTORY).resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Log successful media access
        auth_type = "API key" if auth_info == "api_key" else f"JWT token (room: {auth_info[:20]}...)"
        logger.info(f"📁 Media download authorized: {filename} (auth: {auth_type})")
        
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
