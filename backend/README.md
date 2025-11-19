# FPL AI Optimizer - Backend

FastAPI backend with ML models and optimization engine.

## Structure

```
backend/
├── src/
│   ├── api/              # FastAPI application
│   │   ├── main.py      # Main app with route registration
│   │   └── routes/      # Modular route handlers
│   ├── core/            # Core business logic
│   │   ├── config.py    # Configuration management
│   │   ├── database.py  # Database layer
│   │   ├── optimizer.py # Optimization engine
│   │   └── data_collector.py
│   ├── models/          # ML Models
│   │   ├── ensemble_predictor.py
│   │   ├── points_model.py
│   │   └── minutes_model.py
│   ├── utils/           # Analysis utilities
│   ├── services/        # External services
│   └── schemas/         # Pydantic models
├── tests/               # Test suite
├── data/                # Database & saved models
└── logs/                # Application logs
```

## Running the Backend

```bash
# Development mode
python -m backend.src.api.main

# Or with uvicorn
uvicorn backend.src.api.main:app --reload --port 8000

# Production
uvicorn backend.src.api.main:app --host 0.0.0.0 --port 8000
```

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

## Environment Variables

Create a `.env` file in the backend directory:

```env
DATABASE_PATH=data/fpl_data.db
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000
```

## Testing

```bash
pytest
pytest --cov=src tests/
```

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- FastAPI - Web framework
- scikit-learn, XGBoost, LightGBM - ML
- OR-Tools - Mathematical optimization
- pandas, numpy - Data processing

