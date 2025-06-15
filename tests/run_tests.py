#!/usr/bin/env python3
"""
Test runner script for boo_memories project
This script can be run inside the Docker container to execute all tests
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path

def setup_test_environment():
    """Setup test environment variables and directories"""
    # Set test environment variables
    os.environ["API_KEY"] = "test_api_key_12345"
    os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./tests/test_chat_database.db"
    os.environ["MEDIA_DIRECTORY"] = "./tests/test_media_files"
    os.environ["ENVIRONMENT"] = "test"
    
    # Create test directories
    Path("./tests/test_media_files").mkdir(exist_ok=True)
    
    print("✅ Test environment setup complete")

def run_tests():
    """Run the test suite"""
    try:
        # Setup environment
        setup_test_environment()
        
        # Change to the parent directory (project root)
        os.chdir(Path(__file__).parent.parent)
        
        # Run pytest with coverage
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/test_main.py",
            "-v",
            "--tb=short",
            "--asyncio-mode=auto",
            "--cov=main",
            "--cov-report=term-missing",
            "--cov-report=html:coverage/htmlcov"
        ]
        
        print(f"🧪 Running tests with coverage: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("✅ All tests passed!")
            print("📊 Coverage report generated in coverage/htmlcov/ directory")
            print("📋 Coverage summary displayed above")
        else:
            print(f"❌ Tests failed with return code: {result.returncode}")
            
        return result.returncode
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)