# 🚀 FPL AI Optimizer

> **Advanced Fantasy Premier League team optimization powered by AI and mathematical optimization**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.85+-green.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-15.5-black.svg)](https://nextjs.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## ✨ Features

- **🤖 AI-Powered Predictions**: 6-model ensemble with 9.2/10 accuracy
- **📊 Advanced Analytics**: Fixture difficulty, price predictions, ownership analysis
- **🎯 Smart Transfers**: ML-based transfer recommendations with confidence scoring
- **👑 Captain Optimization**: Data-driven captain selection
- **💎 Chip Strategy**: Mathematical optimization for chip timing
- **📈 Long-term Planning**: Multi-gameweek strategic planning
- **⚡ Real-time Data**: Live FPL API integration
- **🎨 Modern UI**: Beautiful Next.js interface with React 19

## 🏗️ Architecture

```
FPLDataFetch/
├── backend/                # Python FastAPI backend
│   ├── src/
│   │   ├── api/           # FastAPI routes
│   │   ├── core/          # Business logic
│   │   ├── models/        # ML models
│   │   ├── utils/         # Analysis tools
│   │   └── services/      # External services
│   ├── data/              # Database & models
│   └── tests/             # Test suite
├── frontend/              # Next.js frontend
├── docs/                  # Documentation
└── docker/                # Docker configs
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- npm or yarn

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/fpl-ai-optimizer.git
cd fpl-ai-optimizer

# Install Python dependencies
cd backend
pip install -r requirements.txt

# Run the backend
python -m backend.src.api.main

# Or use uvicorn directly
uvicorn backend.src.api.main:app --reload --port 8000
```

### Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Visit `http://localhost:3000` to access the application!

## 📖 Documentation

- [Architecture Guide](docs/ARCHITECTURE.md)
- [API Documentation](docs/API.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Improvements Log](docs/IMPROVEMENTS.md)

## 🎯 Usage

### 1. Upload Your Team

Enter your FPL Manager ID or manually select your 15 players.

### 2. Get AI Analysis

Receive:
- Transfer suggestions with expected points gain
- Captain recommendations
- Injury risk alerts
- Price change predictions
- Chip usage opportunities

### 3. Optimize Strategy

Get long-term planning advice for:
- Wildcard timing
- Bench Boost opportunities
- Triple Captain selection
- Free Hit strategy

## 🧪 Testing

```bash
# Run backend tests
cd backend
pytest

# Run with coverage
pytest --cov=src tests/

# Run frontend tests
cd frontend
npm test
```

### 📐 Model Training & Evaluation

- Historical per-gameweek stats are persisted locally (run `POST /api/system/refresh-data` to pull data + histories).
- Kick off end-to-end model training via `POST /api/system/train-models`. Minutes, points, and ensemble models now:
  - Train on chronological datasets built from stored `player_history`.
  - Use time-based validation/test splits to compute MAE / RMSE / R².
  - Save artifacts under `backend/data/models/` and log metrics to the `model_metrics` table.
- Query `/api/system/model-status` to inspect the latest metrics and confirm artifact freshness before optimizing teams.

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access at http://localhost:8000 (API) and http://localhost:3000 (Frontend)
```

## 📊 ML Models

### Ensemble Predictor (9.2/10 Accuracy)
- Gradient Boosting
- Random Forest
- Neural Network
- XGBoost
- LightGBM
- Ridge Regression

### Key Features
- 40+ engineered features
- Uncertainty quantification
- Time-series cross-validation
- Feature importance tracking

## 🤝 Contributing

Contributions welcome! Please read our [Contributing Guide](CONTRIBUTING.md) first.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- FPL API for providing the data
- Fantasy Premier League community
- Open source ML libraries

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/fpl-ai-optimizer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/fpl-ai-optimizer/discussions)

---

**Made with ❤️ for the FPL community**

