import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv
from environment import LoanUnderwritingEnv, Action

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
API_KEY = os.getenv("API_KEY")
BENCHMARK = "loan-underwriting"
TASKS = ["task_easy", "task_medium", "task_hard", "task_batch"]
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


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
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        if task_id == "task_batch":
            done = False
            step_num = 0
            final_score = 0.0

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

                obs, reward, done, info = env.step(action)
                step_num += 1
                steps_taken = step_num

                step_reward = reward.score if done else 0.0
                rewards.append(step_reward)

                action_str = f"{action.decision}(amount={action.approved_amount:.0f},rate={action.interest_rate})"
                log_step(step=step_num, action=action_str, reward=step_reward, done=done)

                if done:
                    final_score = reward.score
                    score = final_score

            success = score >= 0.5

        else:
            prompt = build_prompt(obs.applicant)
            parsed = get_decision(prompt)
            action = Action(
                decision=parsed.get("decision", "reject"),
                approved_amount=float(parsed.get("approved_amount", 0.0)),
                interest_rate=float(parsed.get("interest_rate", 0.0)),
                reason=parsed.get("reason", "")
            )

            obs, reward, done, info = env.step(action)
            steps_taken = 1
            score = reward.score
            rewards.append(score)
            success = score >= 0.5

            action_str = f"{action.decision}(amount={action.approved_amount:.0f},rate={action.interest_rate})"
            log_step(step=1, action=action_str, reward=score, done=done)

    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        log_end(success=False, steps=steps_taken, score=0.0, rewards=rewards)
        return 0.0

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


def main():
    all_scores = {}
    for task_id in TASKS:
        print(f"\n{'='*40}", flush=True)
        print(f"Running {task_id}...", flush=True)
        print(f"{'='*40}", flush=True)
        score = run_task(task_id)
        all_scores[task_id] = score

    print(f"\n{'='*40}", flush=True)
    print(f"FINAL SCORES: {all_scores}", flush=True)
    print(f"AVERAGE: {sum(all_scores.values())/len(all_scores):.3f}", flush=True)


if __name__ == "__main__":
    main()