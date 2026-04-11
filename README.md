---
title: Loan Underwriting Env
emoji: 💰
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
---

# Loan Underwriting Environment

An reinforcement learning environment where an AI agent acts as a loan underwriting officer. The agent evaluates real-world applicant profiles and makes financial decisions — approve, reject, or counter-offer — while managing portfolio risk and avoiding predatory lending.

Built for the Meta OpenEnv Hackathon.

## Why This Environment?

Loan underwriting is a $100B+ problem. Banks and fintechs use ML models to evaluate creditworthiness, but training and evaluating these agents requires realistic, structured environments. This environment fills that gap — providing a rigorous, graded RL environment for financial decision-making agents.

## Tasks

| Task | Difficulty | Description |
|------|------------|-------------|
| `task_easy` | Easy | Strong applicant (credit > 740, DTI < 0.25). Agent must approve with correct interest rate (5–8%). |
| `task_medium` | Medium | Borderline applicant (credit 600–680, DTI 0.30–0.45). Agent must counter-offer with reduced amount. |
| `task_hard` | Hard | High-risk applicant (credit < 620, DTI > 0.45). Agent must reject or counter-offer with strong reasoning. |
| `task_batch` | Hard | Portfolio task — 3 applicants, $100k capital pool. Maximize approvals while managing risk. |

## Observation Space
```json
{
  "applicant_id": "string",
  "age": "integer",
  "annual_income": "float",
  "credit_score": "integer",
  "debt_to_income_ratio": "float",
  "employment_years": "float",
  "loan_amount_requested": "float",
  "loan_purpose": "string",
  "task_id": "string",
  "difficulty": "string",
  "message": "string"
}
```

## Action Space
```json
{
  "decision": "approve | reject | counter_offer",
  "approved_amount": "float",
  "interest_rate": "float",
  "reason": "string"
}
```

## Reward Function

- Partial credit for correct risk assessment
- Penalized for approving predatory loan purposes (crypto, gambling)
- Penalized for exceeding capital pool in batch task
- Bonus for good reasoning on hard cases
- Score range: 0.01 – 0.99

## Baseline Scores

| Task | Score |
|------|-------|
| task_easy | 0.99 |
| task_medium | 0.99 |
| task_hard | 0.99 |
| task_batch | 0.99 |
| **Average** | **0.99** |

## Setup
```bash
pip install -r requirements.txt
```

## Run Baseline
```bash
python inference.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/reset` | Start new episode |
| POST | `/step` | Take an action |
| GET | `/state` | Current state |
| GET | `/tasks` | List all tasks |

## Environment Variables
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.3-70b-versatile
HF_TOKEN=your_huggingface_token
## Docker
```bash
docker build -t loan-underwriting-env .
docker run -p 7860:7860 loan-underwriting-env
```
