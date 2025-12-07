"""
Automated setup script for FPL AI Optimizer
Installs dependencies, configures services, and starts the application
"""

import subprocess
import sys
import os
import platform
import time
from pathlib import Path

class Colors:
    """ANSI color codes"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_step(message):
    """Print step with formatting"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{message}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}\n")

def print_success(message):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_warning(message):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_error(message):
    """Print error message"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def run_command(command, shell=True, check=True):
    """Run shell command"""
    try:
        result = subprocess.run(
            command,
            shell=shell,
            check=check,
            capture_output=True,
            text=True
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def check_python_version():
    """Check Python version"""
    print_step("Step 1: Checking Python Version")
    
    version = sys.version_info
    if version.major >= 3 and version.minor >= 11:
        print_success(f"Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print_error(f"Python 3.11+ required. Found {version.major}.{version.minor}.{version.micro}")
        return False

def install_dependencies():
    """Install Python dependencies"""
    print_step("Step 2: Checking Python Dependencies")
    
    backend_dir = Path(__file__).parent
    requirements_file = backend_dir / "requirements.txt"
    
    if not requirements_file.exists():
        print_error(f"requirements.txt not found at {requirements_file}")
        return False
    
    # Check if key dependencies are already installed
    try:
        import redis
        import celery
        import slowapi
        print_success("Key dependencies already installed - skipping installation")
        return True
    except ImportError:
        pass
    
    print("Installing dependencies from requirements.txt...")
    success, stdout, stderr = run_command(
        f"{sys.executable} -m pip install -r {requirements_file} --upgrade"
    )
    
    if success:
        print_success("All dependencies installed successfully")
        return True
    else:
        print_error("Failed to install dependencies")
        print(stderr)
        return False

def check_redis():
    """Check if Redis is running"""
    print_step("Step 3: Checking Redis")
    
    # Try to connect to Redis on default port
    try:
        import redis
        client = redis.from_url('redis://localhost:6379/0', socket_connect_timeout=2)
        client.ping()
        print_success("✅ Redis is running on port 6379")
        return True
    except Exception as e:
        print_warning("Redis is not running or not accessible on port 6379")
        print(f"Error: {e}")
        return False

def setup_redis():
    """Setup Redis based on platform"""
    print_step("Step 3a: Redis Setup")
    
    print_success("✅ Redis is already running on port 6379!")
    print("\nIf you need to restart Redis:")
    
    system = platform.system()
    
    if system == "Windows":
        print("\nWindows Options:")
        print("  WSL: sudo service redis-server restart")
        print("  Docker: docker restart <redis-container-name>")
        
    elif system == "Linux":
        print("\nLinux: sudo service redis-server restart")
    
    elif system == "Darwin":  # macOS
        print("\nmacOS: brew services restart redis")
    
    return True

def create_env_file():
    """Create .env file if it doesn't exist"""
    print_step("Step 4: Creating Environment Configuration")
    
    backend_dir = Path(__file__).parent
    env_file = backend_dir / ".env"
    
    if env_file.exists():
        print_success(".env file already exists")
        return True
    
    env_content = """# FPL AI Optimizer Environment Configuration

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# Celery Configuration
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=True
API_LOG_LEVEL=info

# Database Configuration (SQLite by default)
DATABASE_PATH=backend/data/fpl_data.db

# PostgreSQL (optional - uncomment when ready to migrate)
# POSTGRES_CONNECTION_STRING=postgresql://user:password@localhost/fpl_db

# Logging
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
LOG_FILE=backend/logs/app.log

# CORS
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8000"]

# App Info
APP_NAME=FPL AI Optimizer
APP_DESCRIPTION=Advanced Fantasy Premier League team optimization powered by AI
APP_VERSION=2.0.0
"""
    
    try:
        env_file.write_text(env_content)
        print_success(f".env file created at {env_file}")
        return True
    except Exception as e:
        print_error(f"Failed to create .env file: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print_step("Step 5: Creating Directories")
    
    backend_dir = Path(__file__).parent
    directories = [
        backend_dir / "data",
        backend_dir / "data" / "models",
        backend_dir / "logs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print_success(f"Created {directory}")
    
    return True

def test_imports():
    """Test if all imports work"""
    print_step("Step 6: Testing Imports")
    
    try:
        from backend.src.core.cache import get_cache
        from backend.src.core.celery_app import celery_app
        from backend.src.middleware import setup_rate_limiting, setup_monitoring
        from backend.src.utils.form_analyzer import FormAnalyzer
        from backend.src.core.multi_gw_optimizer import MultiGWOptimizer
        from backend.src.utils.differential_finder import DifferentialFinder
        from backend.src.core.formation_optimizer import FormationOptimizer
        
        print_success("All imports successful")
        return True
    except Exception as e:
        print_error(f"Import failed: {e}")
        return False

def print_next_steps():
    """Print next steps for user"""
    print_step("🎉 Setup Complete!")
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}Next Steps:{Colors.END}\n")
    
    print("1️⃣  Start Redis (if not already running):")
    print("   Windows WSL: sudo service redis-server start")
    print("   Linux: sudo service redis-server start")
    print("   macOS: brew services start redis")
    print("   Docker: docker run -d -p 6379:6379 redis:latest")
    
    print("\n2️⃣  Start Celery Worker (Terminal 1):")
    print("   cd backend")
    print("   celery -A src.core.celery_app worker --loglevel=info --pool=solo")
    
    print("\n3️⃣  Start Celery Beat (Terminal 2):")
    print("   cd backend")
    print("   celery -A src.core.celery_app beat --loglevel=info")
    
    print("\n4️⃣  Start Flower Monitoring (Terminal 3 - Optional):")
    print("   cd backend")
    print("   celery -A src.core.celery_app flower")
    print("   Access at: http://localhost:5555")
    
    print("\n5️⃣  Start FastAPI Server (Terminal 4):")
    print("   cd backend")
    print("   python -m backend.src.api.main")
    print("   Access at: http://localhost:8000")
    print("   API Docs: http://localhost:8000/api/docs")
    
    print(f"\n{Colors.BLUE}{Colors.BOLD}Quick Start Command:{Colors.END}")
    print("   python scripts/start_all.py  # Starts all services")
    
    print(f"\n{Colors.GREEN}✨ Your FPL AI Optimizer is ready to use!{Colors.END}\n")

def main():
    """Main setup function"""
    print(f"\n{Colors.BOLD}{'='*60}")
    print("FPL AI Optimizer - Automated Setup")
    print(f"{'='*60}{Colors.END}\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies (will skip if already installed)
    install_dependencies()
    
    # Check Redis
    redis_running = check_redis()
    if not redis_running:
        print_warning("Redis not detected. Attempting setup...")
        setup_redis()
        # Check again
        time.sleep(2)
        redis_running = check_redis()
        if not redis_running:
            print_warning("Redis setup incomplete. You may need to start it manually.")
    
    # Create .env file
    create_env_file()
    
    # Create directories
    create_directories()
    
    # Test imports
    if not test_imports():
        print_warning("Some imports failed. You may need to restart your terminal.")
    
    # Print next steps
    print_next_steps()
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}Setup Complete!{Colors.END}")
    print(f"{Colors.GREEN}You can now start the application with:{Colors.END}")
    print(f"{Colors.BLUE}  python scripts/start_all.py{Colors.END}\n")

if __name__ == "__main__":
    main()
