# Boo Memories

FastAPI backend for secure Matrix chat message and media storage with PIN authentication.

## 🚀 Quick Start

### SQLite (Development)
```bash
cd boo_memories/
mkdir -p data media_files logs
# Create .env file (see Configuration section)
docker-compose --profile sqlite up --build -d
curl http://localhost:8000/health  # Check health
docker-compose --profile sqlite down  # Clean up
```

### PostgreSQL (Production)
```bash
docker-compose --profile postgres up --build -d
```

## ✨ Features

- **Dual Authentication**: API key (for bots) + PIN-based room access (for frontends)
- **Message Storage**: Matrix chat messages with room ID, sender, event ID, content
- **Media Management**: File uploads with automatic linking to messages
- **Database Agnostic**: SQLite (dev) / PostgreSQL (prod) support
- **Auto Cleanup**: Configurable message/media retention policies
- **Real-time Stats**: Database statistics and storage monitoring
- **Security**: API key authentication + JWT-based room tokens

## 🔐 PIN Authentication System

### How It Works
1. **Bot generates PIN**: Using API key → `POST /rooms/{room_id}/pin`
2. **Frontend validates PIN**: No API key needed → `POST /internal/auth/pin`
3. **Returns JWT token**: 24-hour room access token
4. **Frontend uses token**: Access room data via `/ui/rooms/` endpoints

### PIN Security Features
- **6-digit numeric PINs** generated using NIST quantum randomness
- **24-hour expiration** (configurable)
- **Rate limiting**: 3 requests per hour per room
- **One-time use**: PINs become invalid after successful authentication
- **Room-specific**: Each room gets unique PINs

## 🔗 API Endpoints

### Public Endpoints (No Auth Required)
- `GET /health` - API health check
- `POST /internal/auth/pin` - Validate PIN and get room token

### API Key Required (Bot Access)
- `POST /rooms/{room_id}/pin` - Generate PIN for room
- `POST /messages` - Store messages
- `POST /media/upload` - Upload media files
- `GET /stats` - Database statistics
- `POST /cleanup` - Trigger cleanup

### JWT Token Required (Frontend Access)
- `GET /ui/rooms/{room_id}/messages` - Get room messages
- `GET /ui/rooms/{room_id}/stats` - Get room statistics
- `GET /media/{filename}` - Download media files

## ⚙️ Configuration

Create `.env` file in `boo_memories/`:

```bash
# Core API Configuration
API_KEY="your_generated_secure_api_key"
DATABASE_URL="sqlite+aiosqlite:///./data/chat_database.db"
MEDIA_DIRECTORY="/app/media"

# Storage Limits
MAX_DATABASE_SIZE_MB=1000
MAX_FILE_SIZE_MB=50

# CORS Configuration
ALLOWED_ORIGINS="http://localhost:3000,https://your-dashboard-domain.com"
ENVIRONMENT="development"

# PIN Authentication (NEW)
PIN_AUTH_ENABLED=true
PIN_EXPIRY_HOURS=24
PIN_CLEANUP_INTERVAL_HOURS=1
PIN_RATE_LIMIT_PER_HOUR=3
JWT_SECRET_KEY="your_jwt_secret_key_here"
JWT_ALGORITHM="HS256"
JWT_ACCESS_TOKEN_EXPIRE_HOURS=24
```

### Generate Secure Keys
```bash
# Generate API key
python -c "import secrets; print('API_KEY=' + secrets.token_urlsafe(32))"

# Generate JWT secret
python -c "import secrets; print('JWT_SECRET_KEY=' + secrets.token_urlsafe(32))"
```

## 🧪 Testing

```bash
# Build test image
docker build -t boo-memories-test .

# Run all tests (32 tests, 51% coverage)
docker run --rm boo-memories-test python tests/run_tests.py

# Run specific test categories
docker run --rm boo-memories-test python -m pytest tests/test_pin_auth.py -v
docker run --rm boo-memories-test python -m pytest tests/test_api.py -v
```

**Current Status**: ✅ 32 tests passing, 51% coverage

## 🏗️ Architecture

```
boo_memories/
├── main.py                 # FastAPI application
├── config.yaml            # Application configuration
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container definition
├── docker-compose.yml     # Multi-profile deployment
├── tests/                 # Test suites
│   ├── run_tests.py       # Test runner
│   ├── test_api.py        # API endpoint tests
│   └── test_pin_auth.py   # PIN authentication tests
└── data/                  # SQLite database (created at runtime)
```

## 🔗 Integration Flow

```
Matrix Room → boo_bot (API key) → boo_memories API
                ↓
         Generates 6-digit PIN
                ↓
Frontend (PIN only) → boo_memories → JWT room token
                ↓
         Access room messages via /ui/ endpoints
```

## 📚 Technologies

- **FastAPI**: Modern Python web framework
- **SQLAlchemy**: ORM with async support
- **JWT**: Secure room access tokens
- **Docker**: Containerized deployment
- **PostgreSQL/SQLite**: Flexible database backends
- **NIST Beacon**: Quantum randomness for PIN generation

## 🔧 Database Profiles

- **SQLite Profile**: `docker-compose --profile sqlite up -d`
  - Development/testing
  - Single file database
  - No external dependencies

- **PostgreSQL Profile**: `docker-compose --profile postgres up -d`
  - Production deployment
  - Robust concurrent access
  - Automatic backups support