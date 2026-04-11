import random
import math
from pydantic import BaseModel
from typing import Optional, List

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
        self.batch_applicants = []
        self.batch_index = 0
        self.capital_pool = 0.0
        self.batch_actions = []

    def close(self):
        """No-op close for compatibility with evaluator lifecycle hooks."""
        return None

    def _sanitize_action(self, action: Action) -> tuple[Action, str | None]:
        issues = []
        decision = action.decision
        approved_amount = float(action.approved_amount)
        interest_rate = float(action.interest_rate)
        reason = action.reason or ""

        allowed_decisions = {"approve", "reject", "counter_offer"}
        if decision not in allowed_decisions:
            issues.append(f"invalid decision '{decision}', defaulted to reject")
            decision = "reject"
        if not math.isfinite(approved_amount):
            issues.append("approved_amount is non-finite, defaulted to 0")
            approved_amount = 0.0

        if not math.isfinite(interest_rate):
            issues.append("interest_rate is non-finite, defaulted to 0")
            interest_rate = 0.0

        if approved_amount < 0:
            issues.append("approved_amount < 0, clamped to 0")
            approved_amount = 0.0

        if interest_rate < 0:
            issues.append("interest_rate < 0, clamped to 0")
            interest_rate = 0.0

        sanitized = Action(
            decision=decision,
            approved_amount=approved_amount,
            interest_rate=interest_rate,
            reason=reason
        )
        return sanitized, ("; ".join(issues) if issues else None)

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
        self.batch_actions = []
        self.batch_index = 0

        if task_id == "task_batch":
            self.batch_applicants = [
                self._generate_applicant("easy"),
                self._generate_applicant("medium"),
                self._generate_applicant("hard")
            ]
            self.capital_pool = 100000.0
            self.current_applicant = self.batch_applicants[0]
            return Observation(
                applicant=self.current_applicant,
                task_id=task_id,
                difficulty="hard",
                message="You have $100,000 capital pool. Evaluate 3 applicants and optimize approvals while managing portfolio risk. Current applicant 1 of 3."
            )

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
        action, last_action_error = self._sanitize_action(action)

        if self.current_task == "task_batch":
            self.batch_actions.append(action)
            self.batch_index += 1

            if self.batch_index < len(self.batch_applicants):
                self.current_applicant = self.batch_applicants[self.batch_index]
                obs = Observation(
                    applicant=self.current_applicant,
                    task_id=self.current_task,
                    difficulty="hard",
                    message=f"Applicant {self.batch_index + 1} of 3. Capital remaining: ${self.capital_pool:,.2f}"
                )
                return obs, Reward(score=0.01, feedback="Intermediate step"), False, {"steps": self.steps, "last_action_error": last_action_error}
            else:
                reward = self._grade_batch()
                self.done = True
                obs = Observation(
                    applicant=self.current_applicant,
                    task_id=self.current_task,
                    difficulty="hard",
                    message="Batch evaluation complete."
                )
                return obs, reward, self.done, {"steps": self.steps, "last_action_error": last_action_error}

        reward = self._grade(action)
        self.done = True
        obs = Observation(
            applicant=self.current_applicant,
            task_id=self.current_task,
            difficulty=self.current_task.replace("task_", ""),
            message=f"Decision made: {action.decision}"
        )
        return obs, reward, self.done, {"steps": self.steps, "last_action_error": last_action_error}

    def state(self) -> dict:
        return {
            "current_task": self.current_task,
            "done": self.done,
            "steps": self.steps,
            "batch_index": self.batch_index,
            "capital_pool": self.capital_pool,
            "applicant": self.current_applicant.model_dump() if self.current_applicant else None
        }

    def _grade_batch(self) -> Reward:
        score = 0.0
        feedback = []
        total_approved = 0.0
        risky_approvals = 0
        good_decisions = 0

        for i, (applicant, action) in enumerate(zip(self.batch_applicants, self.batch_actions)):
            difficulty = ["easy", "medium", "hard"][i]

            if difficulty == "easy" and action.decision == "approve":
                good_decisions += 1
                total_approved += action.approved_amount
            elif difficulty == "medium" and action.decision in ["approve", "counter_offer"]:
                good_decisions += 1
                total_approved += action.approved_amount
            elif difficulty == "hard" and action.decision == "reject":
                good_decisions += 1
            elif difficulty == "hard" and action.decision == "approve":
                risky_approvals += 1
                total_approved += action.approved_amount

            if applicant.loan_purpose in ["crypto", "gambling_debt"] and action.decision == "approve":
                risky_approvals += 1
                feedback.append(f"Penalized: approved predatory loan purpose ({applicant.loan_purpose})")

        decision_score = good_decisions / 3
        score += decision_score * 0.5
        feedback.append(f"Good decisions: {good_decisions}/3")

        if total_approved <= 100000.0:
            score += 0.3
            feedback.append("Capital pool managed within limits.")
        else:
            feedback.append("Capital pool exceeded — over-approved.")

        if risky_approvals == 0:
            score += 0.2
            feedback.append("No predatory or risky approvals — excellent risk management.")
        else:
            feedback.append(f"Risky approvals detected: {risky_approvals}")

        return Reward(score=min(max(score, 0.01), 0.99), feedback=" | ".join(feedback))

    def _grade(self, action: Action) -> Reward:
        score = 0.01
        feedback = []
        applicant = self.current_applicant
        difficulty = self.current_task.replace("task_", "")

        if applicant.loan_purpose in ["crypto", "gambling_debt"] and action.decision == "approve":
            return Reward(score=0.01, feedback="Penalized: approved predatory loan purpose.")

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

        return Reward(score=min(max(score, 0.01), 0.99), feedback=" | ".join(feedback))
