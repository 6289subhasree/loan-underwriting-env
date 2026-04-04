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

An RL environment where an AI agent acts as a loan underwriting officer, evaluating applicant profiles and making approve/reject/counter-offer decisions.

## Tasks

- **Easy:** Clear-cut applicant with strong financials
- **Medium:** Borderline applicant with mixed signals  
- **Hard:** High-risk applicant requiring rejection or counter-offer

## API Endpoints

- `POST /reset` — Start new episode
- `POST /step` — Take an action
- `GET /state` — Current state
- `GET /tasks` — List all tasks

## Setup
```bash
pip install -r requirements.txt
python inference.py
```

## Baseline Scores

- task_easy: 1.0
- task_medium: 1.0
- task_hard: 1.0
- average: 1.0