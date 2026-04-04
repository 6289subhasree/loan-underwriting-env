import random
from pydantic import BaseModel
from typing import Optional

class Applicant(BaseModel):
    applicant_id: str
    age: int
    annual_income: float
    credit_score: int
    debt_to_income_ratio: float
    employment_years: float
    loan_amount_requested: float
    loan_purpose: str

class Observation(BaseModel):
    applicant: Applicant
    task_id: str
    difficulty: str
    message: str

class Action(BaseModel):
    decision: str
    approved_amount: float
    interest_rate: float
    reason: str

class Reward(BaseModel):
    score: float
    feedback: str

class LoanUnderwritingEnv:
    def __init__(self):
        self.current_applicant = None
        self.current_task = None
        self.done = False
        self.steps = 0
        self.max_steps = 3

    def _generate_applicant(self, difficulty: str) -> Applicant:
        if difficulty == "easy":
            return Applicant(
                applicant_id=f"APP_{random.randint(1000,9999)}",
                age=random.randint(30, 50),
                annual_income=random.uniform(80000, 150000),
                credit_score=random.randint(740, 850),
                debt_to_income_ratio=round(random.uniform(0.10, 0.25), 2),
                employment_years=round(random.uniform(5.0, 20.0), 1),
                loan_amount_requested=random.uniform(10000, 30000),
                loan_purpose=random.choice(["home_improvement", "car", "education"])
            )
        elif difficulty == "medium":
            return Applicant(
                applicant_id=f"APP_{random.randint(1000,9999)}",
                age=random.randint(24, 35),
                annual_income=random.uniform(35000, 60000),
                credit_score=random.randint(600, 680),
                debt_to_income_ratio=round(random.uniform(0.30, 0.45), 2),
                employment_years=round(random.uniform(1.0, 4.0), 1),
                loan_amount_requested=random.uniform(25000, 45000),
                loan_purpose=random.choice(["debt_consolidation", "medical", "home_improvement"])
            )
        else:
            return Applicant(
                applicant_id=f"APP_{random.randint(1000,9999)}",
                age=random.randint(20, 28),
                annual_income=random.uniform(20000, 38000),
                credit_score=random.randint(500, 620),
                debt_to_income_ratio=round(random.uniform(0.45, 0.65), 2),
                employment_years=round(random.uniform(0.2, 1.5), 1),
                loan_amount_requested=random.uniform(40000, 70000),
                loan_purpose=random.choice(["business", "crypto", "gambling_debt"])
            )

    def reset(self, task_id: str = "task_easy") -> Observation:
        self.done = False
        self.steps = 0
        self.current_task = task_id
        difficulty = task_id.replace("task_", "")
        self.current_applicant = self._generate_applicant(difficulty)
        return Observation(
            applicant=self.current_applicant,
            task_id=task_id,
            difficulty=difficulty,
            message="Review this loan application and make a decision."
        )

    def step(self, action: Action) -> tuple:
        self.steps += 1
        reward = self._grade(action)
        self.done = True
        obs = Observation(
            applicant=self.current_applicant,
            task_id=self.current_task,
            difficulty=self.current_task.replace("task_", ""),
            message=f"Decision made: {action.decision}"
        )
        return obs, reward, self.done, {"steps": self.steps}

    def state(self) -> dict:
        return {
            "current_task": self.current_task,
            "done": self.done,
            "steps": self.steps,
            "applicant": self.current_applicant.model_dump() if self.current_applicant else None
        }

    def _grade(self, action: Action) -> Reward:
        score = 0.0
        feedback = []
        applicant = self.current_applicant
        difficulty = self.current_task.replace("task_", "")

        if difficulty == "easy":
            if action.decision == "approve":
                score += 0.5
                feedback.append("Correct decision to approve.")
            else:
                feedback.append("Should have approved — strong profile.")

            if 5.0 <= action.interest_rate <= 8.0:
                score += 0.3
                feedback.append("Interest rate is appropriate.")
            else:
                feedback.append("Interest rate out of expected range (5-8%).")

            if action.approved_amount <= applicant.loan_amount_requested:
                score += 0.2
                feedback.append("Approved amount is reasonable.")

        elif difficulty == "medium":
            if action.decision in ["approve", "counter_offer"]:
                score += 0.4
                feedback.append("Reasonable decision for borderline profile.")
            else:
                feedback.append("Rejection too harsh for this profile.")

            if 8.0 <= action.interest_rate <= 14.0:
                score += 0.3
                feedback.append("Interest rate reflects risk appropriately.")
            else:
                feedback.append("Interest rate doesn't reflect medium risk.")

            if action.approved_amount <= applicant.loan_amount_requested * 0.8:
                score += 0.3
                feedback.append("Reduced amount shows good risk management.")

        else:
            if action.decision == "reject":
                score += 0.5
                feedback.append("Correct rejection of high-risk applicant.")
            elif action.decision == "counter_offer":
                score += 0.3
                feedback.append("Counter offer acceptable but risky.")
            else:
                feedback.append("Approving this profile is very risky.")

            if len(action.reason) > 20:
                score += 0.3
                feedback.append("Good reasoning provided.")

            if action.interest_rate >= 15.0 or action.decision == "reject":
                score += 0.2
                feedback.append("Appropriate handling of hard case.")

        return Reward(score=min(score, 1.0), feedback=" | ".join(feedback))