# Text Summarization using Hugging Face Transformers

## Overview

This project is an end-to-end NLP application for abstractive dialogue summarization using the Pegasus transformer model from Hugging Face. The system generates concise and meaningful summaries from long conversational texts.

The project follows a complete machine learning pipeline architecture including data ingestion, transformation, model training, evaluation, API development, Docker containerization, and cloud deployment.

The application is deployed using Docker and Render for scalable inference and easy accessibility.

---

# Features

- End-to-end NLP pipeline implementation
- Dialogue summarization using Pegasus Transformer
- Modular project architecture
- YAML-based configuration management
- Model training and evaluation pipeline
- FastAPI integration for inference
- Docker containerization
- Cloud deployment using Render
- Batch prediction support

---

# Project Workflow

1. Update `config.yaml`
2. Update `params.yaml`
3. Create Configuration Entity
4. Create Configuration Manager
5. Implement Components
   - Data Ingestion
   - Data Validation
   - Data Transformation
   - Model Trainer
   - Model Evaluation
6. Build Pipelines
   - Training Pipeline
   - Prediction Pipeline
7. Develop FastAPI Application
8. Dockerize the Application
9. Push Docker Image to Docker Hub
10. Deploy using Render

---

# Tech Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| NLP Framework | Hugging Face Transformers |
| Model | Pegasus |
| API Framework | FastAPI |
| Containerization | Docker |
| Deployment | Render |
| Configuration | YAML |

---

# Model Performance

| Metric | Score |
|---|---|
| Loss | 1.5218 |
| ROUGE-1 | 45.3045 |
| ROUGE-2 | 21.7601 |
| ROUGE-L | 35.8643 |
| ROUGE-LSum | 41.6595 |

---

# Project Structure

```bash
dialogue-summarization-system/
│
├── config/
│   └── config.yaml
│
├── params.yaml
├── main.py
├── app.py
├── Dockerfile
├── requirements.txt
├── setup.py
│
├── src/
│   └── textSummarizer/
│       ├── components/
│       ├── config/
│       ├── constants/
│       ├── entity/
│       ├── pipeline/
│       └── utils/
│
├── artifacts/
├── logs/
├── research/


```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Rabilkhan786/dialogue-summarization-system.git
```

## Create Virtual Environment

```bash
conda create -n summarizer python=3.10 -y
conda activate summarizer
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Training Pipeline

```bash
python main.py
```

---

# Run FastAPI Application

```bash
python app.py
```

---

# Docker Setup

## Build Docker Image

```bash
docker build -t text-summarizer .
```

## Run Docker Container

```bash
docker run -p 8080:8080 text-summarizer
```

---

# Deployment

The application was containerized using Docker and deployed on Render.

Deployment Workflow:
1. Build Docker Image
2. Push Docker Image to Docker Hub
3. Pull Docker Image in Render
4. Deploy Application as a Web Service

---

# API Endpoints

## Home Endpoint

```bash
GET /
```

## Train Model Endpoint

```bash
GET /train
```

## Prediction Endpoint

```bash
POST /predict
```

---

# Sample Input

```text
Person A: Hello, how are you?
Person B: I am doing well. I completed the project today.
Person A: Great work. Did you deploy it?
Person B: Yes, I deployed it using Docker and Render.
```

# Sample Output

```text
Person B completed and deployed the project using Docker and Render.
```

---

# Deployment Links

## GitHub Repository

https://github.com/Rabilkhan786/dialogue-summarization-system

## Docker Hub

https://hub.docker.com/

## Live Application

```bash
https://your-app-name.onrender.com
```

---

# Future Improvements

- Add frontend interface
- Improve model performance with larger datasets
- Add authentication and user management
- Implement real-time summarization
- Deploy using Kubernetes

---

# Author

Rabil Khan

GitHub:  
https://github.com/Rabilkhan786