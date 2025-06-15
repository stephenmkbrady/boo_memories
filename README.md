# Boo Memories

This repository contains the `Boo Memories` project, a proof-of-concept FastAPI backend service designed to securely store and manage Matrix chat messages and media files. It provides a robust API for message storage, media uploads, and database management, including cleanup functionalities.

**Note:** This project is currently a development proof of concept and is not production-ready.

## Features

*   **Secure API:** Built with FastAPI, featuring API key authentication for secure access.
*   **Message Storage:** Stores Matrix chat messages with details like room ID, sender, event ID, and content.
*   **Media Management:** Handles media file uploads, linking them to messages, and provides endpoints for media retrieval.
*   **Database Agnostic:** Supports both SQLite (default) and PostgreSQL for data persistence using SQLAlchemy.
*   **Asynchronous Operations:** Leverages `asyncio` and `aiohttp` for high-performance, non-blocking I/O.
*   **Automatic Cleanup:** Includes functionality to clean up old messages and associated media files to manage database size.
*   **Database Statistics:** Provides endpoints to retrieve real-time statistics on messages, media files, and total storage size.

## Getting Started

These instructions will guide you through setting up and running the Boo Memories API using Docker and Docker Compose.

### Prerequisites

*   **Docker**: Ensure Docker is installed and running on your system.
    *   [Install Docker Engine](https://docs.docker.com/engine/install/)
*   **Docker Compose**: Ensure Docker Compose is installed.
    *   [Install Docker Compose](https://docs.docker.com/compose/install/)
*   **Python 3.9+** and `pip` (for generating API key)

### Setup Steps

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/stephenmkbrady/boo_memories.git
    cd boo_memories
    ```

2.  **Create `requirements.txt`:**

    Ensure you have a `requirements.txt` file in the project root. If not, create it with the necessary Python dependencies. Refer to the `requirements.txt` file in this directory for the full list of dependencies.

3.  **Create `Dockerfile`:**

    Ensure you have a `Dockerfile` in the project root. This file defines the Docker image for the API. Refer to the `Dockerfile` in this directory for its content.

4.  **Create `docker-compose.yml`:**

    Ensure you have a `docker-compose.yml` file in the project root. This file defines the Docker services for the API and its database. Refer to the `docker-compose.yml` file in this directory for its content.

5.  **Create necessary directories:**

    ```bash
    mkdir -p data media_files logs
    ```

6.  **Generate API Key and create `.env` file:**

    Generate a strong API key:

    ```bash
    python -c "import secrets; print(secrets.token_urlsafe(32))"
    ```

    Create a `.env` file in the project root with the following content. Replace `your_generated_api_key_here` with the key you just generated.

    ```
    API_KEY="your_generated_api_key_here"
    DATABASE_URL="sqlite+aiosqlite:///./data/chat_database.db"
    MEDIA_DIRECTORY="/app/media"
    MAX_DATABASE_SIZE_MB=1000
    MAX_FILE_SIZE_MB=50
    ALLOWED_ORIGINS="http://localhost:3000,http://192.168.1.100:3000"
    ENVIRONMENT="development"
    ```

    *   **Optional Configuration:**
        *   `DATABASE_URL`: Change to a PostgreSQL connection string if you want to use PostgreSQL (e.g., `postgresql+asyncpg://chatuser:chatpass@postgres:5432/chatdb`).
        *   `MEDIA_DIRECTORY`: Path to store uploaded media files (default: `/app/media` inside container, mapped to `./media_files` on host).
        *   `MAX_DATABASE_SIZE_MB`: Maximum target size for the database in MB (default: 1000).
        *   `MAX_FILE_SIZE_MB`: Maximum allowed size for uploaded media files in MB (default: 50).
        *   `ALLOWED_ORIGINS`: Comma-separated list of origins for CORS.
        *   `ENVIRONMENT`: Set to `development` for more permissive CORS settings.

7.  **Build and start Docker Compose services:**

    Choose your database backend:

    *   **For SQLite (recommended for development):**
        ```bash
        docker-compose --profile sqlite build --no-cache
        docker-compose --profile sqlite up -d
        ```
    *   **For PostgreSQL (recommended for production):**
        ```bash
        docker-compose --profile postgres build --no-cache
        docker-compose --profile postgres up -d
        ```

    The API will be accessible at `http://localhost:8000`.

8.  **Fix media permissions (if uploads fail):**

    If you encounter issues with media uploads, it might be due to file permissions within the Docker container. You can fix this manually:

    *   **For SQLite:**
        ```bash
        docker exec chat-api-sqlite chown -R 1000:1000 /app/media
        docker exec chat-api-sqlite chmod -R 775 /app/media
        ```
    *   **For PostgreSQL:**
        ```bash
        docker exec chat-api-postgres chown -R 1000:1000 /app/media
        docker exec chat-api-postgres chmod -R 775 /app/media
        ```
    After fixing permissions, you might need to restart the service:
    ```bash
    docker-compose --profile <your_profile> restart
    ```

9.  **Verify the setup:**

    Check if the API is running and responsive:

    ```bash
    curl http://localhost:8000/health
    ```

    You should see a JSON response indicating `{"status": "healthy"}`.

## API Endpoints

*   `/health`: Check API health and version.
*   `/messages`: GET to retrieve messages for a room; POST to store a new message.
*   `/messages/{message_id}`: GET a specific message; DELETE a message.
*   `/media/upload`: Upload a media file and link it to a message.
*   `/media/{filename}`: Download a media file.
*   `/stats`: Get database statistics.
*   `/cleanup`: Trigger database cleanup based on age or size.

## Testing

The project includes a comprehensive test suite with 19 tests covering all major functionality. Tests are designed to run inside the Docker container to ensure consistency with the production environment.

### Test Coverage

- **Total Tests**: 32 ✅
- **Test Coverage**: 51% (166/325 statements)
- **All Tests Passing**: ✅

### Covered Functionality

- ✅ All API endpoints (`/health`, `/messages`, `/media`, `/stats`, `/cleanup`)
- ✅ Message CRUD operations
- ✅ Media upload/download functionality
- ✅ Database operations and statistics
- ✅ Authentication and input validation
- ✅ Error handling for common scenarios

### Running Tests

#### Using Docker (Recommended)

Run the complete test suite with coverage reporting inside the Docker container:

```bash
# Build the test image
docker build -t boo-memories-test .

# Run tests with coverage report
docker run --rm boo-memories-test python tests/run_tests.py
```

This will output:
- Test results
- Coverage summary with missing line numbers
- HTML coverage report (generated in container)

### Test Structure

```
tests/
├── test_main.py              # Main test suite (32 comprehensive tests)
├── run_tests.py              # Test runner with coverage reporting
├── test_chat_database.db     # Test database file
├── test_media_files/         # Test media directory
└── __pycache__/              # Python cache

coverage/
├── COVERAGE_REPORT.md        # Detailed coverage analysis
└── htmlcov/                  # HTML coverage reports (generated)
```

### Test Categories

1. **Core API Tests**
   - Health check endpoint
   - Message creation, retrieval, and deletion
   - Database statistics

2. **Media Handling Tests**
   - File upload and validation
   - Media download and streaming
   - Media cleanup with message deletion

3. **Validation & Error Handling**
   - Input validation for all endpoints
   - Authentication testing
   - Error responses for invalid requests

4. **Edge Cases & Advanced Features**
   - Duplicate event ID handling
   - Custom timestamp support
   - Cleanup dry-run functionality

### Coverage Report

For detailed coverage analysis, see [`coverage/COVERAGE_REPORT.md`](coverage/COVERAGE_REPORT.md) which includes:
- Line-by-line coverage breakdown
- Recommendations for improving coverage
- Analysis of covered vs uncovered functionality

### Test Environment

Tests use:
- **Isolated test database**: `tests/test_chat_database.db`
- **Separate media directory**: `tests/test_media_files/`
- **Mocked authentication**: For consistent test execution
- **Async test framework**: `pytest-asyncio` for proper async testing

## Technologies Used

*   [FastAPI](https://fastapi.tiangolo.com/) - Web framework for building APIs
*   [SQLAlchemy](https://www.sqlalchemy.org/) - SQL toolkit and Object Relational Mapper
*   [Uvicorn](https://www.uvicorn.org/) - ASGI server
*   [aiohttp](https://aiohttp.readthedocs.io/) - Asynchronous HTTP client/server
*   [aiofiles](https://github.com/Tinche/aiofiles) - Asynchronous file operations
*   [Pydantic](https://pydantic-docs.helpmanual.io/) - Data validation and settings management
*   [python-dotenv](https://pypi.org/project/python-dotenv/) - For managing environment variables
*   [python-jose](https://python-jose.readthedocs.io/) - For JSON Web Tokens (JWT)
*   [passlib](https://passlib.readthedocs.io/) - For password hashing
*   [asyncpg](https://magicstack.github.io/asyncpg/current/) - PostgreSQL driver for asyncio
*   [aiosqlite](https://aiosqlite.readthedocs.io/) - SQLite driver for asyncio
*   [psycopg2-binary](https://pypi.org/project/psycopg2-binary/) - PostgreSQL adapter for Python

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details (if applicable).

## Acknowledgements

This project was partially generated by Claude 4 Sonnet and Gemini 2.5 Flash.
