"""
Start all FPL AI Optimizer services
Starts Redis, Celery Worker, Celery Beat, and FastAPI server
"""

import subprocess
import sys
import os
import time
import platform
import requests
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header():
    """Print startup header"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}")
    print("🚀 Starting FPL AI Optimizer Services")
    print(f"{'='*60}{Colors.END}\n")

def check_redis():
    """Check if Redis is running"""
    try:
        import redis
        client = redis.from_url('redis://localhost:6379/0', socket_connect_timeout=2)
        client.ping()
        print(f"{Colors.GREEN}✅ Redis is running{Colors.END}")
        return True
    except:
        print(f"{Colors.RED}❌ Redis is not running{Colors.END}")
        return False

def start_redis():
    """Start Redis based on platform"""
    system = platform.system()
    
    print(f"{Colors.YELLOW}⚠️  Starting Redis...{Colors.END}")
    
    if system == "Linux":
        subprocess.Popen(["sudo", "service", "redis-server", "start"])
    elif system == "Darwin":  # macOS
        subprocess.Popen(["brew", "services", "start", "redis"])
    else:  # Windows
        print(f"{Colors.YELLOW}Please start Redis manually:{Colors.END}")
        print("  WSL: sudo service redis-server start")
        print("  Docker: docker run -d -p 6379:6379 redis:latest")
        return False
    
    time.sleep(2)
    return check_redis()

def start_celery_worker():
    """Start Celery worker"""
    print(f"\n{Colors.BLUE}Starting Celery Worker...{Colors.END}")
    
    project_root = Path(__file__).parent.parent
    
    # Windows requires --pool=solo
    pool_arg = "--pool=solo" if platform.system() == "Windows" else ""
    
    cmd = f"celery -A backend.src.tasks.celery_app worker --loglevel=info -Q high_priority,medium_priority,low_priority,default {pool_arg}"
    
    # Don't suppress output - let it show in console
    process = subprocess.Popen(
        cmd,
        shell=True,
        cwd=project_root
    )
    
    print(f"{Colors.GREEN}✅ Celery Worker started (PID: {process.pid}){Colors.END}")
    return process

def start_celery_beat():
    """Start Celery Beat scheduler"""
    print(f"\n{Colors.BLUE}Starting Celery Beat...{Colors.END}")
    
    project_root = Path(__file__).parent.parent
    
    cmd = "celery -A backend.src.tasks.celery_app beat --loglevel=info"
    
    # Don't suppress output - let it show in console
    process = subprocess.Popen(
        cmd,
        shell=True,
        cwd=project_root
    )
    
    print(f"{Colors.GREEN}✅ Celery Beat started (PID: {process.pid}){Colors.END}")
    return process

def start_flower():
    """Start Flower monitoring UI"""
    print(f"\n{Colors.BLUE}Starting Flower (Celery Monitoring)...{Colors.END}")
    
    project_root = Path(__file__).parent.parent
    
    cmd = "celery -A backend.src.tasks.celery_app flower"
    
    # Don't suppress output - let it show in console
    process = subprocess.Popen(
        cmd,
        shell=True,
        cwd=project_root
    )
    
    print(f"{Colors.GREEN}✅ Flower started (PID: {process.pid}){Colors.END}")
    print(f"{Colors.BLUE}   Access at: http://localhost:5555{Colors.END}")
    return process

def wait_for_api_ready(max_attempts=30, delay=1):
    """Wait for FastAPI to be ready by polling health endpoint"""
    print(f"{Colors.YELLOW}⏳ Waiting for API to be ready...{Colors.END}")
    
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                print(f"{Colors.GREEN}✅ API is ready!{Colors.END}")
                return True
        except requests.exceptions.RequestException:
            pass
        
        if attempt < max_attempts - 1:
            time.sleep(delay)
    
    print(f"{Colors.RED}❌ API failed to become ready after {max_attempts} seconds{Colors.END}")
    return False

def start_fastapi():
    """Start FastAPI server"""
    print(f"\n{Colors.BLUE}Starting FastAPI Server...{Colors.END}")
    
    backend_dir = Path(__file__).parent.parent / "backend"
    
    cmd = f"{sys.executable} -m backend.src.api.main"
    
    # Don't suppress output - let it show in console
    process = subprocess.Popen(
        cmd,
        shell=True,
        cwd=backend_dir.parent
    )
    
    print(f"{Colors.GREEN}✅ FastAPI Server started (PID: {process.pid}){Colors.END}")
    print(f"{Colors.BLUE}   API: http://localhost:8000{Colors.END}")
    print(f"{Colors.BLUE}   Docs: http://localhost:8000/api/docs{Colors.END}")
    
    # Wait for API to be ready
    if not wait_for_api_ready():
        print(f"{Colors.YELLOW}⚠️  API may not be fully ready, but continuing...{Colors.END}")
    
    return process

def print_summary(processes):
    """Print summary of running services"""
    print(f"\n{Colors.GREEN}{Colors.BOLD}{'='*60}")
    print("✅ All Services Started Successfully!")
    print(f"{'='*60}{Colors.END}\n")
    
    print(f"{Colors.BOLD}Running Services:{Colors.END}")
    for name, process in processes.items():
        print(f"  • {name}: PID {process.pid}")
    
    print(f"\n{Colors.BOLD}Access Points:{Colors.END}")
    print("  • API: http://localhost:8000")
    print("  • API Docs: http://localhost:8000/api/docs")
    print("  • Flower: http://localhost:5555")
    print("  • Health Check: http://localhost:8000/health/detailed")
    
    print(f"\n{Colors.YELLOW}Press Ctrl+C to stop all services{Colors.END}\n")

def main():
    """Main startup function"""
    print_header()
    
    # Check/start Redis
    if not check_redis():
        if not start_redis():
            print(f"{Colors.RED}Cannot start without Redis. Exiting.{Colors.END}")
            sys.exit(1)
    
    processes = {}
    
    try:
        # Start services
        processes['Celery Worker'] = start_celery_worker()
        time.sleep(2)
        
        processes['Celery Beat'] = start_celery_beat()
        time.sleep(1)
        
        processes['Flower'] = start_flower()
        time.sleep(1)
        
        processes['FastAPI'] = start_fastapi()
        
        # Print summary
        print_summary(processes)
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Shutting down services...{Colors.END}")
        
        # Terminate all processes
        for name, process in processes.items():
            try:
                process.terminate()
                print(f"{Colors.GREEN}✅ Stopped {name}{Colors.END}")
            except:
                pass
        
        print(f"\n{Colors.GREEN}All services stopped. Goodbye!{Colors.END}\n")
        sys.exit(0)

if __name__ == "__main__":
    main()
