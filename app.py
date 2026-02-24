# app.py
"""
Intelligent Medical Diagnosis & Treatment Recommendation System
==============================================================
Entry point for the multi-agent medical AI pipeline.

Usage:
    python app.py
    python app.py --image scans/chest_xray.png
"""

import argparse
import json
import os
from graph.medical_graph import build_medical_graph


def print_section(title: str, content: str, emoji: str = "📋"):
    width = 65
    print("\n" + "=" * width)
    print(f"  {emoji}  {title}")
    print("=" * width)
    print(content)


def run_diagnosis(patient_data: dict, image_path: str = None) -> dict:
    """
    Run the full multi-agent medical diagnosis pipeline.

    Args:
        patient_data: Dict of patient vitals and lab values
        image_path:   Optional path to chest X-ray / scan image

    Returns:
        Final pipeline state with diagnosis, treatment, and report
    """
    graph = build_medical_graph()

    initial_state = {
        "patient_data": patient_data,
        "image_path": image_path,
        "risk_assessment": None,
        "image_analysis": None,
        "final_diagnosis": None,
        "treatment_plan": None,
        "patient_report": None,
        "messages": []
    }

    print("\n🏥  Starting Medical AI Diagnosis Pipeline...")
    print(f"   Patient Profile: {json.dumps(patient_data, indent=6)}")
    if image_path:
        print(f"   Imaging: {image_path}")

    result = graph.invoke(initial_state)

    # ── Print Results ──
    print_section("Risk Assessment (Scikit-learn)", json.dumps(result["risk_assessment"], indent=2), "🔬")
    if result.get("image_analysis"):
        print_section("Imaging Analysis (TensorFlow)", json.dumps(result["image_analysis"], indent=2), "🩻")
    print_section("Final Diagnosis", result["final_diagnosis"], "🩺")
    print_section("Treatment Plan (LangChain RAG)", result["treatment_plan"], "💊")
    print_section("Patient Report", result["patient_report"], "📄")

    print("\n✅  Pipeline complete.\n")
    return result


def main():
    parser = argparse.ArgumentParser(description="Medical AI Diagnosis System")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to chest X-ray image (optional)")
    parser.add_argument("--age",          type=int,   default=52)
    parser.add_argument("--bmi",          type=float, default=28.4)
    parser.add_argument("--glucose",      type=int,   default=145)
    parser.add_argument("--bp",           type=int,   default=92,  dest="blood_pressure")
    parser.add_argument("--cholesterol",  type=int,   default=220)
    parser.add_argument("--smoking",      type=int,   default=1,   choices=[0, 1])
    parser.add_argument("--family-history", type=int, default=1,   choices=[0, 1],
                        dest="family_history")
    args = parser.parse_args()

    patient_data = {
        "age":            args.age,
        "bmi":            args.bmi,
        "glucose":        args.glucose,
        "blood_pressure": args.blood_pressure,
        "cholesterol":    args.cholesterol,
        "smoking":        args.smoking,
        "family_history": args.family_history,
    }

    run_diagnosis(patient_data, image_path=args.image)


if __name__ == "__main__":
    # Quick demo without CLI args
    demo_patient = {
        "age": 52,
        "bmi": 28.4,
        "glucose": 145,
        "blood_pressure": 92,
        "cholesterol": 220,
        "smoking": 1,
        "family_history": 1
    }
    run_diagnosis(demo_patient)
