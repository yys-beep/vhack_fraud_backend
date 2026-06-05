# Fraud Shield Backend

A FastAPI-based machine learning API for fraud detection and analysis. This backend provides intelligent fraud detection capabilities using XGBoost and scikit-learn models, integrated with a React frontend for comprehensive fraud case management.

## 🎯 Project Overview

**Fraud Shield Backend** is the core intelligence system for the vhack fraud detection initiative. This API:
- Runs trained ML models (XGBoost, scikit-learn) for fraud prediction
- Processes and analyzes fraud case data
- Provides RESTful endpoints for real-time fraud detection
- Manages historical fraud data and analytics
- Deployed on Render for production use

## 📋 Tech Stack

| Category | Technologies |
|----------|--------------|
| **Web Framework** | FastAPI |
| **ASGI Server** | Uvicorn |
| **ML Libraries** | XGBoost, scikit-learn |
| **Data Processing** | Pandas |
| **Data Validation** | Pydantic |
| **Model Persistence** | joblib |
| **Language** | Python |

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip or conda
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yys-beep/vhack_fraud_backend.git
cd vhack_fraud_backend
```

2. Create and activate a virtual environment:
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n fraud-backend python=3.10
conda activate fraud-backend
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Development

Start the development server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

Access the interactive API documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📁 Project Structure

```
vhack_fraud_backend/
├── main.py                 # FastAPI application entry point
├── models/                 # Trained ML models (XGBoost, scikit-learn)
│   ├── fraud_detector.pkl
│   └── scaler.pkl
├── routes/                 # API endpoints
│   ├── predictions.py
│   └── analytics.py
├── schemas/                # Pydantic data models
│   └── fraud_case.py
├── services/               # Business logic
│   └── fraud_detection.py
├── requirements.txt        # Python dependencies
└── README.md
```

## 🔑 Key Features

- **Real-time Fraud Detection**: Process transactions and predict fraud probability
- **XGBoost Models**: High-performance gradient boosting for accurate predictions
- **scikit-learn Integration**: Robust preprocessing and feature scaling
- **REST API**: Clean, documented endpoints for frontend integration
- **Data Validation**: Pydantic models ensure data integrity
- **Interactive Docs**: Auto-generated Swagger/OpenAPI documentation
- **Scalable**: Ready for production deployment with Uvicorn + Gunicorn

## 📦 Dependencies

```
fastapi              # Modern web framework for APIs
uvicorn              # ASGI server
pandas               # Data manipulation and analysis
xgboost              # Gradient boosting library
scikit-learn         # Machine learning utilities
pydantic             # Data validation
joblib               # Model serialization
```

For complete list, see `requirements.txt`

## 🔌 API Endpoints

### Fraud Prediction
```
POST /api/predict
```
Predict fraud probability for a transaction

**Request:**
```json
{
  "transaction_id": "txn_123",
  "amount": 150.00,
  "merchant_id": "mch_456",
  "user_id": "usr_789"
}
```

**Response:**
```json
{
  "fraud_probability": 0.87,
  "is_fraud": true,
  "confidence": "high",
  "explanation": "Transaction amount exceeds usual pattern"
}
```

### Analytics
```
GET /api/analytics/summary
```
Get fraud analytics and statistics

```
GET /api/cases
```
Retrieve fraud cases and history

## 🌐 Deployment on Render

This backend is designed to be deployed on [Render](https://render.com).

### Deployment Steps

1. **Connect GitHub Repository**
   - Go to Render Dashboard
   - Click "Create +" → "Web Service"
   - Select this repository

2. **Configure Environment**
   - Environment: Python 3.10
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

3. **Set Environment Variables**
   - Add any required API keys or configuration variables
   - Database URLs (if applicable)
   - Secret keys for JWT/authentication

4. **Deploy**
   - Click "Deploy"
   - Monitor logs for successful startup
   - Your API will be available at `https://your-service-name.onrender.com`

### Health Check
```bash
curl https://your-service-name.onrender.com/health
```

## 🔗 Architecture & Related Projects

### Frontend Integration
Connects to **[vhack_fraud_frontend](https://github.com/yys-beep/vhack_fraud_frontend)** - React-based web application that:
- Displays fraud case dashboards
- Shows real-time predictions
- Generates PDF reports
- Manages fraud investigations

**Frontend Tech Stack:**
- React 19.2.0
- Vite (build tool)
- Firebase (data storage)
- jsPDF (reporting)

### Data Flow
```
React Frontend
    ↓ (HTTP Requests)
FastAPI Backend (Render)
    ↓ (ML Processing)
XGBoost + scikit-learn Models
    ↓ (Predictions)
Response back to Frontend
```

## 📚 Resources & References

### Exploratory Data Analysis (EDA)
Comprehensive analysis of the fraud detection dataset with visualization and preprocessing:
- **Google Colab Notebook**: [EDA & Data Preprocessing](https://colab.research.google.com/drive/1BP2KAiXkH02Ln1xZw34iODUkB1BPK57g?usp=sharing)
  - Data exploration and structure analysis
  - Statistical analysis and visualizations
  - Data cleaning and feature engineering
  - Preprocessing pipeline documentation

## 🧪 Testing

Run tests with pytest:
```bash
pytest tests/
```

## 🔒 Security Considerations

- Use environment variables for sensitive data
- Implement rate limiting for API endpoints
- Add authentication/authorization (JWT recommended)
- Validate all input data with Pydantic
- Use HTTPS in production (Render provides this)
- Keep dependencies updated regularly

## 📊 Model Management

### Training New Models
```python
# Update model files in models/ directory
# Retrain with latest data and export as .pkl files
import joblib
joblib.dump(trained_model, 'models/fraud_detector.pkl')
```

### Model Versioning
- Keep previous model versions for rollback
- Document model performance metrics
- Test new models before deployment

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Submit a Pull Request

## 📝 License

This project is currently unlicensed. Please contact the repository owner for licensing information.

## 📞 Support & Documentation

For issues, questions, or contributions:
1. Check existing [GitHub Issues](https://github.com/yys-beep/vhack_fraud_backend/issues)
2. Open a new issue with detailed information
3. Submit a pull request with improvements
4. Check FastAPI docs: `http://localhost:8000/docs` (during development)

## 🔗 Quick Links

- **Frontend Repository**: [vhack_fraud_frontend](https://github.com/yys-beep/vhack_fraud_frontend)
- **Backend Repository**: [vhack_fraud_backend](https://github.com/yys-beep/vhack_fraud_backend)
- **Render Documentation**: [render.com/docs](https://render.com/docs)
- **FastAPI Documentation**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com)
- **XGBoost Documentation**: [xgboost.readthedocs.io](https://xgboost.readthedocs.io)

## 🚀 Production Checklist

- [ ] Environment variables configured on Render
- [ ] Database connections tested
- [ ] Authentication/authorization implemented
- [ ] Rate limiting enabled
- [ ] Error handling and logging configured
- [ ] CORS settings configured for frontend
- [ ] Health check endpoint operational
- [ ] Models tested and validated
- [ ] Monitoring and alerts set up
- [ ] Backup strategy for models and data

---

**Repository**: [yys-beep/vhack_fraud_backend](https://github.com/yys-beep/vhack_fraud_backend)

**Created**: March 11, 2026

**Last Updated**: March 19, 2026
