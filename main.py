from fastapi import FastAPI
from environment import LoanUnderwritingEnv, Action
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Loan Underwriting Environment")
env = LoanUnderwritingEnv()

@app.get("/")
def root():
    return {"message": "Loan Underwriting Environment is running!"}

@app.post("/reset")
def reset(task_id: str = "task_easy"):
    obs = env.reset(task_id=task_id)
    return obs

@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    return env.state()

@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "task_id": "task_easy",
                "difficulty": "easy",
                "description": "Clear-cut applicant with strong financials. Approve with appropriate rate."
            },
            {
                "task_id": "task_medium",
                "difficulty": "medium",
                "description": "Borderline applicant with mixed signals. Weigh risk carefully."
            },
            {
                "task_id": "task_hard",
                "difficulty": "hard",
                "description": "High-risk applicant. Reject or counter-offer with strong reasoning."
            }
        ]
    }