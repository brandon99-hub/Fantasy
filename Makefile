# FPL AI Optimizer - Makefile
# Common commands for development and deployment

.PHONY: help install run-backend run-frontend run-all test clean docker-build docker-up docker-down

help:
	@echo "FPL AI Optimizer - Available commands:"
	@echo ""
	@echo "  make install        - Install all dependencies"
	@echo "  make run-backend    - Run backend server"
	@echo "  make run-frontend   - Run frontend development server"
	@echo "  make run-all        - Run both backend and frontend"
	@echo "  make test           - Run all tests"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make docker-build   - Build Docker images"
	@echo "  make docker-up      - Start Docker containers"
	@echo "  make docker-down    - Stop Docker containers"
	@echo ""

install:
	@echo "Installing backend dependencies..."
	cd backend && pip install -r requirements.txt
	@echo "Installing frontend dependencies..."
	cd frontend && npm install
	@echo "✅ All dependencies installed!"

run-backend:
	@echo "Starting backend server..."
	python -m backend.src.api.main

run-frontend:
	@echo "Starting frontend development server..."
	cd frontend && npm run dev

test:
	@echo "Running backend tests..."
	cd backend && pytest
	@echo "✅ All tests passed!"

clean:
	@echo "Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf backend/.pytest_cache
	rm -rf frontend/.next
	rm -rf frontend/out
	@echo "✅ Cleaned!"

docker-build:
	@echo "Building Docker images..."
	docker-compose build

docker-up:
	@echo "Starting Docker containers..."
	docker-compose up -d
	@echo "✅ Containers started!"
	@echo "Backend: http://localhost:8000"
	@echo "Frontend: http://localhost:3000"

docker-down:
	@echo "Stopping Docker containers..."
	docker-compose down
	@echo "✅ Containers stopped!"

train-models:
	@echo "Training ML models..."
	python scripts/train_models.py

update-data:
	@echo "Updating FPL data..."
	curl -X POST http://localhost:8000/api/system/refresh-data

