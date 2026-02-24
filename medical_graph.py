# graph/medical_graph.py
import json
import os
import joblib
from typing import TypedDict, List, Optional

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage


# ─────────────────────────── State Schema ────────────────────────────────────

class MedicalState(TypedDict):
    patient_data: dict
    image_path: Optional[str]
    risk_assessment: Optional[dict]   # Output from Scikit-learn
    image_analysis: Optional[dict]    # Output from TensorFlow CNN
    final_diagnosis: Optional[str]    # Synthesized by LLM
    treatment_plan: Optional[str]     # From LangChain RAG
    patient_report: Optional[str]     # Patient-friendly summary
    messages: List


# ─────────────────────────── LLM ─────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)


# ─────────────────────────── Agent 1: Diagnosis ───────────────────────────────

def diagnosis_agent(state: MedicalState) -> MedicalState:
    """
    Runs Scikit-learn risk classifier + TensorFlow image analyzer,
    then asks GPT-4o to synthesize both into a final diagnosis.
    """
    from ml_models.risk_classifier import DiseaseRiskClassifier
    from ml_models.image_analyzer import MedicalImageAnalyzer

    # --- Scikit-learn ---
    clf = DiseaseRiskClassifier()
    model_path = "models/risk_classifier.pkl"
    if os.path.exists(model_path):
        clf.load(model_path)
    else:
        # Train on synthetic data if no saved model
        from ml_models.risk_classifier import generate_sample_data
        X, y = generate_sample_data(2000)
        clf.train(X, y)

    risk_result = clf.predict_risk(state["patient_data"])

    # --- TensorFlow (optional) ---
    image_result = None
    if state.get("image_path") and os.path.exists(state["image_path"]):
        analyzer = MedicalImageAnalyzer(weights_path="models/image_model.h5")
        image_result = analyzer.analyze_image(state["image_path"])

    # --- LLM Synthesis ---
    synthesis_prompt = f"""
You are a senior diagnostic physician AI. Synthesize the ML model outputs below
into a final diagnosis with severity level (Mild / Moderate / Severe).

Tabular Risk Assessment (Scikit-learn):
{json.dumps(risk_result, indent=2)}

Imaging Analysis (TensorFlow CNN):
{json.dumps(image_result, indent=2) if image_result else "No imaging provided."}

Patient Demographics & Labs:
{json.dumps(state["patient_data"], indent=2)}

Provide:
- Primary diagnosis
- Severity level
- Key clinical indicators that support this diagnosis
- Confidence narrative
"""
    response = llm.invoke([HumanMessage(content=synthesis_prompt)])

    return {
        **state,
        "risk_assessment": risk_result,
        "image_analysis": image_result,
        "final_diagnosis": response.content,
        "messages": state["messages"] + [
            AIMessage(content=f"[DiagnosisAgent] {response.content}")
        ]
    }


# ─────────────────────────── Agent 2: Treatment Planner ──────────────────────

def treatment_agent(state: MedicalState) -> MedicalState:
    """
    Uses LangChain RAG to fetch evidence-based treatment recommendations.
    Falls back to LLM general knowledge if no knowledge base is found.
    """
    from rag.medical_rag import MedicalRAG

    rag = MedicalRAG()
    try:
        rag.load_knowledge_base()
        treatment = rag.get_treatment_recommendations(
            diagnosis=state["final_diagnosis"],
            patient_info=state["patient_data"]
        )
    except FileNotFoundError:
        # Fallback: direct LLM without RAG
        fallback_prompt = f"""
As a medical AI, provide evidence-based treatment recommendations for:

Diagnosis: {state["final_diagnosis"]}
Patient Profile: {json.dumps(state["patient_data"], indent=2)}

Structure your response with:
1. Immediate Interventions
2. Medications (generic names + dosage ranges)
3. Lifestyle Modifications
4. Follow-up Schedule
5. Red Flag Symptoms
"""
        response = llm.invoke([HumanMessage(content=fallback_prompt)])
        treatment = response.content

    return {
        **state,
        "treatment_plan": treatment,
        "messages": state["messages"] + [
            AIMessage(content="[TreatmentAgent] Treatment plan generated.")
        ]
    }


# ─────────────────────────── Agent 3: Patient Communication ──────────────────

def communication_agent(state: MedicalState) -> MedicalState:
    """
    Translates clinical language into a patient-friendly report.
    """
    report_prompt = f"""
You are a compassionate medical communicator. Create a clear, empathetic patient report.

Diagnosis: {state["final_diagnosis"]}
Risk Scores: {json.dumps(state.get("risk_assessment"), indent=2)}
Imaging: {json.dumps(state.get("image_analysis"), indent=2)}
Treatment Plan: {state["treatment_plan"]}

Write a patient-friendly report (~300 words) that:
- Explains the condition in plain English (no jargon)
- Lists 3–5 key action items as simple steps
- Suggests follow-up timeline
- Uses a warm, reassuring tone
- Closes with 2–3 motivational health goals
"""
    response = llm.invoke([HumanMessage(content=report_prompt)])

    return {
        **state,
        "patient_report": response.content,
        "messages": state["messages"] + [
            AIMessage(content="[CommunicationAgent] Patient report ready.")
        ]
    }


# ─────────────────────────── Emergency Agent ─────────────────────────────────

def emergency_agent(state: MedicalState) -> MedicalState:
    """Triggered when high-confidence critical diagnosis is detected."""
    return {
        **state,
        "patient_report": (
            "⚠️  HIGH RISK ALERT\n\n"
            "Based on your test results, our AI has detected indicators that require "
            "IMMEDIATE medical attention.\n\n"
            f"Detected: {state.get('risk_assessment', {}).get('diagnosis', 'Critical condition')}\n"
            f"Confidence: {state.get('risk_assessment', {}).get('confidence', 'N/A')}%\n\n"
            "👉 Please visit the nearest Emergency Room or call emergency services NOW.\n"
            "Do NOT drive yourself. Contact a family member or call an ambulance."
        ),
        "messages": state["messages"] + [
            AIMessage(content="[EmergencyAgent] CRITICAL: Immediate attention required.")
        ]
    }


# ─────────────────────────── Router ──────────────────────────────────────────

def should_escalate(state: MedicalState) -> str:
    """Route to emergency if high-confidence critical condition detected."""
    risk = state.get("risk_assessment", {})
    confidence = risk.get("confidence", 0)
    diagnosis = risk.get("diagnosis", "")

    if confidence > 85 and any(kw in diagnosis for kw in ["Heart Disease", "Severe"]):
        return "emergency"
    return "treatment"


# ─────────────────────────── Graph Builder ────────────────────────────────────

def build_medical_graph() -> StateGraph:
    """Compile and return the LangGraph multi-agent workflow."""
    graph = StateGraph(MedicalState)

    graph.add_node("diagnosis", diagnosis_agent)
    graph.add_node("treatment", treatment_agent)
    graph.add_node("communication", communication_agent)
    graph.add_node("emergency", emergency_agent)

    graph.set_entry_point("diagnosis")

    graph.add_conditional_edges(
        "diagnosis",
        should_escalate,
        {"treatment": "treatment", "emergency": "emergency"}
    )
    graph.add_edge("treatment", "communication")
    graph.add_edge("communication", END)
    graph.add_edge("emergency", END)

    return graph.compile()
