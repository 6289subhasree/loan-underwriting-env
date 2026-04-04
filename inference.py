import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from environment import LoanUnderwritingEnv, Action
SPACE_URL = "https://subhasreeee-loan-underwriting-env.hf.space"

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("API_BASE_URL")
)

MODEL_NAME = os.getenv("MODEL_NAME")
TASKS = ["task_easy", "task_medium", "task_hard"]

def run_task(task_id: str):
    env = LoanUnderwritingEnv()
    obs = env.reset(task_id=task_id)
    applicant = obs.applicant

    prompt = f"""You are a loan underwriting AI agent. Review this application and make a decision.

Applicant Profile:
- Age: {applicant.age}
- Annual Income: ${applicant.annual_income:,.2f}
- Credit Score: {applicant.credit_score}
- Debt-to-Income Ratio: {applicant.debt_to_income_ratio:.2%}
- Employment Years: {applicant.employment_years}
- Loan Amount Requested: ${applicant.loan_amount_requested:,.2f}
- Loan Purpose: {applicant.loan_purpose}

You must respond in this exact JSON format:
{{
    "decision": "approve" or "reject" or "counter_offer",
    "approved_amount": <float>,
    "interest_rate": <float>,
    "reason": "<your reasoning here>"
}}

Respond with JSON only, no extra text."""

    print(json.dumps({
        "event": "START",
        "task_id": task_id,
        "applicant_id": applicant.applicant_id
    }))

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    raw = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        parsed = json.loads(match.group()) if match else {
            "decision": "reject",
            "approved_amount": 0.0,
            "interest_rate": 0.0,
            "reason": "Could not parse response"
        }

    action = Action(
        decision=parsed.get("decision", "reject"),
        approved_amount=float(parsed.get("approved_amount", 0.0)),
        interest_rate=float(parsed.get("interest_rate", 0.0)),
        reason=parsed.get("reason", "")
    )

    print(json.dumps({
        "event": "STEP",
        "task_id": task_id,
        "action": action.model_dump()
    }))

    obs, reward, done, info = env.step(action)

    print(json.dumps({
        "event": "END",
        "task_id": task_id,
        "score": reward.score,
        "feedback": reward.feedback
    }))

    return reward.score

def main():
    scores = {}
    for task_id in TASKS:
        print(f"\n{'='*40}")
        print(f"Running {task_id}...")
        print(f"{'='*40}")
        score = run_task(task_id)
        scores[task_id] = score

    print(json.dumps({
        "event": "SUMMARY",
        "scores": scores,
        "average": sum(scores.values()) / len(scores)
    }))

if __name__ == "__main__":
    main()