#!/bin/bash
# ============================================================
#  push_to_github.sh
#  Run this once inside the medical-ai-system/ folder to
#  initialize git and push everything to your GitHub repo.
# ============================================================

set -e   # Exit immediately on error

REPO_NAME="medical-ai-system"
GITHUB_USERNAME="VijayKumaro7"
REMOTE_URL="https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " 🚀  Pushing ${REPO_NAME} to GitHub"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Initialize git (safe to run even if already initialized)
git init

# 2. Configure git identity (change these if needed)
git config user.name  "Vijay Kumar"
git config user.email "your_email@example.com"   # ← update this

# 3. Stage all files
git add .

# 4. Initial commit
git commit -m "feat: initial commit — Intelligent Medical AI Diagnosis System

- Scikit-learn GBM risk classifier for tabular patient data
- TensorFlow EfficientNetB0 CNN for chest X-ray classification
- LangChain RAG pipeline over medical literature (ChromaDB)
- LangGraph multi-agent orchestration with emergency escalation
- CLI + Python API entry points
- Comprehensive README with architecture diagrams"

# 5. Rename branch to main
git branch -M main

# 6. Add remote origin
git remote remove origin 2>/dev/null || true
git remote add origin "${REMOTE_URL}"

# 7. Push
echo ""
echo "📡  Pushing to ${REMOTE_URL} ..."
git push -u origin main

echo ""
echo "✅  Done! Visit: https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
