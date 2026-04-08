from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from environment import LoanUnderwritingEnv, Action
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Loan Underwriting Environment")
env = LoanUnderwritingEnv()

# Serve static UI
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/ui")
def ui():
    return FileResponse("static/index.html")


@app.get("/")
def root():
    return FileResponse("static/index.html")


# Reset environment
@app.post("/reset")
def reset(task_id: str = "task_easy"):
    obs = env.reset(task_id=task_id)
    return obs.model_dump()  # ✅ SAFE serialization


# Take a step
@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),   # ✅ SAFE
        "reward": reward.model_dump(),     # ✅ SAFE
        "done": done,
        "info": info
    }


# Get current state
@app.get("/state")
def state():
    return env.state()


# List tasks
@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "task_id": "task_easy",
                "difficulty": "easy",
                "description": "Clear-cut applicant with strong financials. Approve with appropriate rate.",
                "scoring": "approve + interest 5-8% + amount within requested = 1.0"
            },
            {
                "task_id": "task_medium",
                "difficulty": "medium",
                "description": "Borderline applicant with mixed signals. Weigh risk carefully.",
                "scoring": "counter_offer + interest 8-14% + reduced amount = 1.0"
            },
            {
                "task_id": "task_hard",
                "difficulty": "hard",
                "description": "High-risk applicant. Reject or counter-offer with strong reasoning.",
                "scoring": "reject + reasoning + appropriate rate = 1.0"
            },
            {
                "task_id": "task_batch",
                "difficulty": "hard",
                "description": "Portfolio task — evaluate 3 applicants with $100k capital pool. Maximize approvals while managing risk.",
                "scoring": "good decisions + capital management + no predatory approvals = 1.0"
            }
        ]
    }