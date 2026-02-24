# 🏥 Intelligent Medical Diagnosis & Treatment Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange?style=for-the-badge&logo=tensorflow)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4%2B-F7931E?style=for-the-badge&logo=scikit-learn)
![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-1C3C3C?style=for-the-badge)
![LangGraph](https://img.shields.io/badge/LangGraph-0.1%2B-4B9CD3?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A production-grade multi-agent AI system that combines classical ML, deep learning, and LLM-powered reasoning to diagnose diseases and recommend personalized treatments.**

[Features](#-features) • [Architecture](#-architecture) • [Setup](#-setup) • [Usage](#-usage) • [Tech Stack](#-tech-stack) • [Results](#-results)

</div>

---

## 📌 Overview

This project demonstrates a complete end-to-end medical AI pipeline that:

- **Classifies disease risk** from structured patient data (vitals, labs) using a Gradient Boosting classifier (Scikit-learn)
- **Analyzes medical images** (chest X-rays) using a CNN with EfficientNetB0 transfer learning (TensorFlow)
- **Retrieves evidence-based treatments** from indexed medical literature using RAG (LangChain + ChromaDB)
- **Orchestrates all agents** in a conditional workflow with emergency escalation routing (LangGraph)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔬 **Risk Classification** | Gradient Boosting on tabular vitals — age, BMI, glucose, cholesterol, BP |
| 🩻 **Medical Imaging** | EfficientNetB0 CNN classifies chest X-rays: Normal / Pneumonia / TB / COVID-19 |
| 📚 **RAG Knowledge Base** | LangChain indexes clinical PDFs (PubMed, WHO guidelines) into ChromaDB |
| 🤖 **Multi-Agent Workflow** | LangGraph orchestrates Diagnosis → Treatment → Communication agents |
| ⚠️ **Emergency Routing** | Automatic escalation for high-confidence critical diagnoses |
| 📄 **Patient Reports** | GPT-4o translates clinical findings into plain-English patient summaries |

---

## 🏗️ Architecture

```
Patient Input (vitals + optional X-ray)
              │
              ▼
┌─────────────────────────────────────┐
│         LangGraph Orchestrator      │
│                                     │
│  ┌──────────────────────────────┐   │
│  │     Diagnosis Agent          │   │
│  │  ┌─────────┐ ┌────────────┐  │   │
│  │  │Scikit   │ │TensorFlow  │  │   │
│  │  │GBM Risk │ │EfficientNet│  │   │
│  │  │Classifier│ │CNN (X-ray) │  │   │
│  │  └─────────┘ └────────────┘  │   │
│  │         GPT-4o Synthesis      │   │
│  └──────────────────────────────┘   │
│              │                      │
│     ┌────────┴────────┐             │
│     ▼                 ▼             │
│  ┌──────────┐   ┌──────────┐        │
│  │Treatment │   │Emergency │        │
│  │ Agent    │   │ Agent    │        │
│  │(RAG+LLM) │   │(Escalate)│        │
│  └──────────┘   └──────────┘        │
│       │                             │
│       ▼                             │
│  ┌──────────────┐                   │
│  │Communication │                   │
│  │Agent (Report)│                   │
│  └──────────────┘                   │
└─────────────────────────────────────┘
              │
              ▼
    Final Patient Report
```

### Agent Workflow

```
Entry → [DiagnosisAgent]
              │
    ┌─────────┴──────────┐
    │ confidence > 85%   │
    │ + critical disease │
    ▼                    ▼
[Emergency]         [TreatmentAgent]
    │                    │
    └──────────┬─────────┘
               ▼
       [CommunicationAgent]
               │
              END
```

---

## 📁 Project Structure

```
medical-ai-system/
├── app.py                      # Main entry point
├── requirements.txt
├── .env.example
├── .gitignore
│
├── ml_models/
│   ├── risk_classifier.py      # Scikit-learn GBM pipeline
│   └── image_analyzer.py       # TensorFlow EfficientNetB0 CNN
│
├── rag/
│   └── medical_rag.py          # LangChain RAG + ChromaDB
│
├── graph/
│   └── medical_graph.py        # LangGraph multi-agent orchestration
│
├── medical_docs/               # Place your PDFs here (guidelines, papers)
├── models/                     # Saved model weights (.pkl, .h5)
└── scans/                      # Test chest X-ray images
```

---

## ⚙️ Setup

### Prerequisites

- Python 3.10+
- OpenAI API key (for GPT-4o)

### 1. Clone the Repository

```bash
git clone https://github.com/VijayKumaro7/medical-ai-system.git
cd medical-ai-system
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. (Optional) Build Medical Knowledge Base

Place clinical guideline PDFs in `medical_docs/`, then run:

```python
from rag.medical_rag import MedicalRAG
rag = MedicalRAG()
rag.build_knowledge_base("medical_docs/")
```

If no PDFs are provided, the system falls back to GPT-4o's built-in medical knowledge.

---

## 🚀 Usage

### Quick Demo

```bash
python app.py
```

### Custom Patient Data via CLI

```bash
python app.py \
  --age 52 \
  --bmi 28.4 \
  --glucose 145 \
  --bp 92 \
  --cholesterol 220 \
  --smoking 1 \
  --family-history 1 \
  --image scans/chest_xray.png   # Optional
```

### Python API

```python
from app import run_diagnosis

patient = {
    "age": 52,
    "bmi": 28.4,
    "glucose": 145,
    "blood_pressure": 92,
    "cholesterol": 220,
    "smoking": 1,
    "family_history": 1
}

result = run_diagnosis(patient, image_path="scans/chest_xray.png")

print(result["final_diagnosis"])
print(result["treatment_plan"])
print(result["patient_report"])
```

### Train Models Independently

```bash
# Train Scikit-learn classifier on your dataset
python ml_models/risk_classifier.py

# Build/inspect the TensorFlow CNN architecture
python ml_models/image_analyzer.py
```

---

## 📊 Expected Output

```
🏥  Starting Medical AI Diagnosis Pipeline...

═══════════════════════════════════════════════════════════════
  🔬  Risk Assessment (Scikit-learn)
═══════════════════════════════════════════════════════════════
{
  "diagnosis": "Diabetes Risk",
  "confidence": 84.72,
  "risk_scores": {
    "Healthy": 4.10,
    "Diabetes Risk": 84.72,
    "Heart Disease Risk": 7.33,
    "Hypertension Risk": 3.85
  }
}

═══════════════════════════════════════════════════════════════
  🩺  Final Diagnosis
═══════════════════════════════════════════════════════════════
Primary Diagnosis: Type 2 Diabetes Mellitus — Moderate Severity
Key indicators: Fasting glucose 145 mg/dL, BMI 28.4 (overweight),
positive family history, active smoker...

═══════════════════════════════════════════════════════════════
  💊  Treatment Plan
═══════════════════════════════════════════════════════════════
1. Immediate Interventions: Blood glucose monitoring...
2. Medications: Metformin 500mg twice daily...
3. Lifestyle: Mediterranean diet, 150 min/week aerobic exercise...
...

═══════════════════════════════════════════════════════════════
  📄  Patient Report
═══════════════════════════════════════════════════════════════
Your test results show signs of Type 2 Diabetes...
```

---

## 🧠 Tech Stack

| Library | Version | Role |
|---|---|---|
| **Scikit-learn** | ≥1.4.0 | Disease risk classification (Gradient Boosting) |
| **TensorFlow** | ≥2.15.0 | Medical image CNN (EfficientNetB0 transfer learning) |
| **LangChain** | ≥0.2.0 | RAG pipeline over clinical PDFs |
| **LangGraph** | ≥0.1.0 | Multi-agent workflow orchestration |
| **ChromaDB** | ≥0.4.0 | Vector store for document embeddings |
| **OpenAI GPT-4o** | — | LLM for synthesis, planning, and patient communication |
| **Pandas / NumPy** | — | Data manipulation |
| **Pillow** | — | Image preprocessing |

---

## 📈 Model Performance

| Model | Dataset | Accuracy | Notes |
|---|---|---|---|
| GBM Risk Classifier | Synthetic + PIMA diabetes | ~88–92% | Tabular vitals & labs |
| EfficientNetB0 CNN | Chest X-ray (NIH/Kaggle) | ~90%+ | 4-class classification |
| RAG Retrieval | PubMed / WHO PDFs | — | Grounded, source-cited answers |

---

## 🔮 Roadmap

- [ ] FastAPI REST endpoint for deployment
- [ ] Streamlit / Gradio web UI
- [ ] DICOM image support
- [ ] Fine-tune on CheXpert / NIH ChestX-ray14 datasets
- [ ] Add SHAP explainability for ML predictions
- [ ] Docker containerization
- [ ] HL7 FHIR EHR integration

---

## ⚠️ Disclaimer

> This project is for **educational and portfolio purposes only**.
> It is **not a medical device** and should **not be used for clinical decision-making**.
> Always consult a qualified healthcare professional for medical advice.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Vijay Kumar**
- GitHub: [@VijayKumaro7](https://github.com/VijayKumaro7)
- LinkedIn: [linkedin.com/in/vijaykumar](https://linkedin.com/in/vijaykumar)

---

<div align="center">
  <strong>⭐ Star this repo if you found it useful!</strong>
</div>
