import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv
from environment import LoanUnderwritingEnv, Action

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("API_BASE_URL")
)

MODEL_NAME = os.getenv("MODEL_NAME")
TASKS = ["task_easy", "task_medium", "task_hard", "task_batch"]


def get_decision(prompt: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {
            "decision": "reject",
            "approved_amount": 0.0,
            "interest_rate": 0.0,
            "reason": "Could not parse response"
        }


def build_prompt(applicant, context: str = "") -> str:
    return f"""You are a loan underwriting AI agent. Review this application and make a decision.

Applicant Profile:
- Age: {applicant.age}
- Annual Income: ${applicant.annual_income:,.2f}
- Credit Score: {applicant.credit_score}
- Debt-to-Income Ratio: {applicant.debt_to_income_ratio:.2%}
- Employment Years: {applicant.employment_years}
- Loan Amount Requested: ${applicant.loan_amount_requested:,.2f}
- Loan Purpose: {applicant.loan_purpose}

{context}

Rules:
- Never approve crypto or gambling loans
- Approve strong profiles (credit > 720, DTI < 0.30)
- Counter-offer borderline profiles (credit 600-720, DTI 0.30-0.45)
- Reject high-risk profiles (credit < 600, DTI > 0.45)

Respond in this exact JSON format:
{{
    "decision": "approve" or "reject" or "counter_offer",
    "approved_amount": <float>,
    "interest_rate": <float>,
    "reason": "<your reasoning here>"
}}

JSON only, no extra text."""


def run_task(task_id: str) -> float:
    env = LoanUnderwritingEnv()
    obs = env.reset(task_id=task_id)

    if task_id == "task_batch":
        print(json.dumps({
            "event": "START",
            "task_id": task_id,
            "message": "Batch evaluation — 3 applicants, $100k capital pool"
        }))

        done = False
        step_num = 0
        final_score = 0.0
        final_feedback = ""

        while not done:
            prompt = build_prompt(
                obs.applicant,
                context=f"Context: {obs.message}\nOptimize approvals while managing portfolio risk within $100,000 capital pool."
            )

            parsed = get_decision(prompt)

            action = Action(
                decision=parsed.get("decision", "reject"),
                approved_amount=float(parsed.get("approved_amount", 0.0)),
                interest_rate=float(parsed.get("interest_rate", 0.0)),
                reason=parsed.get("reason", "")
            )

            print(json.dumps({
                "event": "STEP",
                "task_id": task_id,
                "step": step_num + 1,
                "action": action.model_dump()
            }))

            obs, reward, done, info = env.step(action)
            step_num += 1

            if done:
                final_score = reward.score
                final_feedback = reward.feedback

        print(json.dumps({
            "event": "END",
            "task_id": task_id,
            "score": final_score,
            "feedback": final_feedback
        }))

        return final_score

    print(json.dumps({
        "event": "START",
        "task_id": task_id,
        "applicant_id": obs.applicant.applicant_id
    }))

    prompt = build_prompt(obs.applicant)
    parsed = get_decision(prompt)

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

    print(f"\n{'='*40}")
    print(json.dumps({
        "event": "SUMMARY",
        "scores": scores,
        "average": round(sum(scores.values()) / len(scores), 4)
    }))


if __name__ == "__main__":
    main()