import os
import json
import re
import math
from openai import OpenAI
from dotenv import load_dotenv
from environment import LoanUnderwritingEnv, Action

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.groq.com/openai/v1"
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK = "loan-underwriting"
TASKS = ["task_easy", "task_medium", "task_hard", "task_batch"]

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


MIN_SCORE = 0.01
MAX_SCORE = 0.99


# Normalize score strictly between (0,1)
def normalize_score(score: float) -> float:
    if not isinstance(score, (int, float)) or not math.isfinite(score):
        return MIN_SCORE
    if score <= 0:
        return MIN_SCORE
    if score >= 1:
        return MAX_SCORE
    return score


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, rewards: list):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


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

        # ✅ SAFE fallback (no zeros)
        return {
            "decision": "reject",
            "approved_amount": 0.01,
            "interest_rate": 0.01,
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

    rewards = []
    steps_taken = 0
    score = MIN_SCORE
    success = False
    terminal_error = None

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        if task_id == "task_batch":
            done = False
            step_num = 0

            while not done:
                prompt = build_prompt(
                    obs.applicant,
                    context=f"Context: {obs.message}\nOptimize approvals while managing portfolio risk within $100,000 capital pool."
                )

                parsed = get_decision(prompt)

                action = Action(
                    decision=parsed.get("decision", "reject"),
                    approved_amount=float(parsed.get("approved_amount", 0.01)),
                    interest_rate=float(parsed.get("interest_rate", 0.01)),
                    reason=parsed.get("reason", "")
                )

                obs, reward, done, info = env.step(action)
                last_action_error = info.get("last_action_error")

                step_num += 1
                steps_taken = step_num

                step_reward = normalize_score(reward.score) if done else 0.01
                rewards.append(step_reward)

                action_str = f"{action.decision}(amount={action.approved_amount:.0f},rate={action.interest_rate})"
                log_step(step=step_num, action=action_str, reward=step_reward, done=done, error=last_action_error)

                if done:
                    score = normalize_score(reward.score)

            success = score >= 0.5

        else:
            prompt = build_prompt(obs.applicant)
            parsed = get_decision(prompt)

            action = Action(
                decision=parsed.get("decision", "reject"),
                approved_amount=float(parsed.get("approved_amount", 0.01)),
                interest_rate=float(parsed.get("interest_rate", 0.01)),
                reason=parsed.get("reason", "")
            )

            obs, reward, done, info = env.step(action)
            last_action_error = info.get("last_action_error")

            steps_taken = 1
            score = normalize_score(reward.score)
            rewards.append(score)
            success = score >= 0.5

            action_str = f"{action.decision}(amount={action.approved_amount:.0f},rate={action.interest_rate})"
            log_step(step=1, action=action_str, reward=score, done=done, error=last_action_error)

    except Exception as e:
        terminal_error = str(e)
        score = MIN_SCORE
        success = False
        if not rewards:
            rewards.append(MIN_SCORE)
        if steps_taken == 0:
            log_step(
                step=1,
                action="error()",
                reward=MIN_SCORE,
                done=True,
                error=terminal_error
            )
            steps_taken = 1
    finally:
        env.close()
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return normalize_score(score)


def main():
    for task_id in TASKS:
        run_task(task_id)


if __name__ == "__main__":
    main()
