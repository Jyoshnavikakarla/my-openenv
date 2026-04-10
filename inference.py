from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import  json, re
from env.environment import EmailEnv

from openai import OpenAI
import os
import os
from openai import OpenAI

print("BASE_URL:", os.environ.get("API_BASE_URL"), flush=True)
print("API_KEY:", os.environ.get("API_KEY"), flush=True)

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"]
)

print("DEBUG: LLM CHECK CALLED", flush=True)

# -------------------------------
# SCHEMA FOR INCOMING ACTION
# -------------------------------
class ActionInput(BaseModel):
    category: str
    priority: str
    response: str

# -------------------------------
# SAFE JSON PARSER
# -------------------------------
def extract_json(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return None

# -------------------------------
# EMAIL AGENT
# -------------------------------
# -------------------------------
# EMAIL AGENT
# -------------------------------
class EmailAgent:
    def act(self, observation):
        """
        Generate action (reply) for given observation (email).
        Determine difficulty based on keywords.
        """
        text = observation.email.lower()

        # Default reward based on task difficulty keywords
        if "easy" in text or "simple" in text or "info" in text:
            difficulty = "easy"
            reward_points = 1 + (2 * os.urandom(1)[0] % 3)  # 1-3
        elif "medium" in text or "issue" in text or "help" in text:
            difficulty = "medium"
            reward_points = 4 + (3 * os.urandom(1)[0] % 4)  # 4-7
        elif "hard" in text or "problem" in text or "urgent" in text:
            difficulty = "hard"
            reward_points = 8 + (2 * os.urandom(1)[0] % 3)  # 8-10
        else:
            difficulty = "easy"
            reward_points = 1 + (2 * os.urandom(1)[0] % 3)  # default 1-3

        # Determine category & response
        if "refund" in text or "charged" in text or "payment" in text:
            category = "billing"
            response_text = "Your refund/payment issue is being handled. ✅ Solved"
        elif "login" in text or "crash" in text or "error" in text:
            category = "technical"
            response_text = "We are fixing the technical issue. ✅ Solved"
        else:
            category = "general"
            response_text = "We will get back to you shortly. ⚠️ Not Solved"

        # Return as action dict
        return {
            "category": category,
            "priority": difficulty,
            "response": response_text,
            "reward_points": reward_points
        }
# -------------------------------
# FASTAPI APP
# -------------------------------
app = FastAPI()

# -------------------------------
# TASK ENVS AND AGENTS
# -------------------------------
envs = {
    "easy": EmailEnv(task="easy"),
    "medium": EmailEnv(task="medium"),
    "hard": EmailEnv(task="hard")
}
agents = {k: EmailAgent() for k in envs.keys()}

# -------------------------------
# TRACK STEP INFO
# -------------------------------
task_stats = {
    k: {
        "step": 0,
        "emails_received": 0,
        "emails_sent": 0,
        "total_reward": 0.0
    } for k in envs.keys()
}

# -------------------------------
# RESET ENDPOINT
# -------------------------------
from fastapi import Request

@app.post("/reset")
async def reset_post(request: Request):
    try:
        data = await request.json()
        task = data.get("task", "easy")
    except:
        # If no body → default task
        task = "easy"

    if task not in envs:
        return {"error": "Invalid task"}

    obs = envs[task].reset()

    # ✅ Example email injected automatically
    obs = obs.dict()
    obs["email"] = {
        "subject": "Login Issue",
        "content": "I can't log into my account even after resetting password.",
        "sender": "user@example.com"
    }

    return {
        "observation": obs,
        "reward": 0.0,
        "done": False
    }

# Put it into observation (if your env supports it)

    
@app.get("/reset/{task}")
def reset(task: str):
    if task not in envs:
        return {"error": "Invalid task"}
    obs = envs[task].reset()
    task_stats[task]["step"] = 0
    task_stats[task]["emails_received"] = 0
    task_stats[task]["emails_sent"] = 0
    task_stats[task]["total_reward"] = 0.0
    return {
        "observation": obs.dict(),
        "task": task,
        "emails_received": 0,
        "emails_sent": 0,
        "reward_points": 0.0
    }
@app.post("/reset")
def reset_post(input: dict):
    task = input.get("task")

    if task not in envs:
        return {"error": "Invalid task"}

    obs = envs[task].reset()

    task_stats[task]["step"] = 0
    task_stats[task]["emails_received"] = 0
    task_stats[task]["emails_sent"] = 0
    task_stats[task]["total_reward"] = 0.0

    return {
        "observation": obs.dict(),
        "reward": 0.0,
        "done": False
    }

# -------------------------------
# STEP ENDPOINT
# -------------------------------

@app.post("/step")
def step_post(input: dict):
    task = input.get("task")

    if task not in envs:
        return {"error": "Invalid task"}

    action_dict = {
        "category": input.get("category"),
        "priority": input.get("priority"),
        "response": input.get("response")
    }

    obs, reward, done, _ = envs[task].step(action_dict)

    return {
        "observation": obs.dict(),
        "reward": float(reward.score),
        "done": done
    }
    
@app.get("/state")
def state():
    return task_stats

# -------------------------------
# QUICK TEST STEP (GET)
# -------------------------------
@app.post("/step/{task}")
def step(task: str, action: ActionInput):
    if task not in envs:
        return {"status": "fail", "reason": "Invalid task"}

    # Convert input
    action_dict = action.dict()

    # -----------------------
    # VALIDATION LAYER
    # -----------------------
    if not validate_action(action_dict):
        return {
            "status": "fail",
            "reason": "Invalid action format"
        }

    # -----------------------
    # LLM CRITERIA CHECK
    # -----------------------
    if not llm_check(action_dict["response"]):
        return {
            "status": "fail",
            "reason": "LLM criteria failed"
        }

    # -----------------------
    # ENV STEP
    # -----------------------
    obs, reward, done, _ = envs[task].step(action_dict)

    reward_score = action_dict.get("reward_points", reward.score)

    stats = task_stats[task]
    stats["step"] += 1
    stats["emails_received"] += 1
    stats["emails_sent"] += 1
    stats["total_reward"] += reward_score

    return {
        "status": "success",
        "task": task,
        "step": stats["step"],
        "emails_received": stats["emails_received"],
        "emails_sent": stats["emails_sent"],
        "reward_points": reward_score,
        "average_reward": round(stats["total_reward"] / stats["step"], 2),
        "resolved": "✅ Solved" in action_dict.get("response", ""),
        "observation": obs.dict(),
        "done": done
    }
print("STEP HIT", flush=True)
# -------------------------------
# HOME ROUTE (SHOW ALL LINKS)
# -------------------------------
@app.get("/")
def home():
    import os

    # Get HF Space URL
    space_host = os.getenv("SPACE_HOST")

    if space_host:
        base_url = f"https://{space_host}"
    else:
        base_url = "http://localhost:7860"

    return {
        "message": "🚀 Email Agent API is running!",
        "endpoints": {
            "easy": {
                "reset": f"{base_url}/reset/easy",
                "step_post": f"{base_url}/step/easy",
                "step_get": f"{base_url}/step/easy"
            },
            "medium": {
                "reset": f"{base_url}/reset/medium",
                "step_post": f"{base_url}/step/medium",
                "step_get": f"{base_url}/step/medium"
            },
            "hard": {
                "reset": f"{base_url}/reset/hard",
                "step_post": f"{base_url}/step/hard",
                "step_get": f"{base_url}/step/hard"
            }
        },
        "docs": f"{base_url}/docs",
        "note": "Use POST /step/{task} with JSON body to send email"
    }


# -------------------------------
# BASELINE SCORING SCRIPT WITH LINKS
# -------------------------------
if __name__ == "__main__":
    import socket

    # Get local IP to use in links
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    port = 8080  # same as uvicorn port

    base_url = f"http://{local_ip}:{port}"

    print("[START] Email Agent Baseline Simulation\n")
    print(f"Reset endpoint: {base_url}/reset/<task>")
    print(f"Step endpoint (POST): {base_url}/step/<task>\n")

    for task_name in ["easy", "medium", "hard"]:
        env = envs[task_name]
        agent = agents[task_name]
        obs = env.reset()
        done = False
        step_id = 0
        total_reward = 0.0
        emails_received = 0
        emails_sent = 0

        print(f"--- Running task: {task_name} ---")
        print(f"Reset link: {base_url}/reset/{task_name}")
        print(f"Step link (POST): {base_url}/step/{task_name}\n")

        while not done:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward.score
            emails_received += 1
            emails_sent += 1
            print(f"[STEP] task={task_name} step={step_id} action={action} reward={reward.score} resolved={reward.score>0}")
            step_id += 1

        avg_reward = round(total_reward / step_id, 2)
        print(f"[TASK BASELINE] task={task_name} average_reward={avg_reward} emails_received={emails_received} emails_sent={emails_sent}\n")

    print("[END] Simulation Completed")
    print(f"Access your API endpoints in browser or via curl/postman at: {base_url}/reset/<task> and {base_url}/step/<task>")
    # -------------------------------
# STARTUP EVENT (HF LOG LINKS)
# -------------------------------
@app.on_event("startup")
def startup_event():
    import os

    # Hugging Face runs on port 7860
    port = os.getenv("PORT", "7860")

    # HF Spaces URL (fallback to localhost if not available)
    space_url = os.getenv("SPACE_HOST")
    
    if space_url:
        base_url = f"https://{space_url}"
    else:
        base_url = f"http://localhost:{port}"

    print("\n==============================")
    print("🚀 Email Agent Started!")
    print("==============================\n")

    print("🔗 Available Endpoints:\n")

    for task in ["easy", "medium", "hard"]:
        print(f"➡️ RESET ({task}): {base_url}/reset/{task}")
        print(f"➡️ STEP  ({task}) [POST]: {base_url}/step/{task}")
        print(f"➡️ STEP  ({task}) [GET ]: {base_url}/step/{task}\n")

    print("💡 Use POST /step/{task} with JSON body to send email")
    print("==============================\n")
def validate_action(action):
    valid_categories = ["billing", "technical", "general"]
    valid_priorities = ["easy", "medium", "hard"]

    if action["category"] not in valid_categories:
        return False
    if action["priority"] not in valid_priorities:
        return False
    if len(action["response"]) < 15:
        return False

    return True
def llm_check(response_text):
    try:
        prompt = f"""
        Check if this email response is professional and helpful:

        "{response_text}"

        Answer ONLY in JSON:
        {{"valid": true or false}}
        """

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        result = extract_json(completion.choices[0].message["content"])
        print("LLM raw response:", completion, flush=True)
        
        return result.get("valid", False) if result else False
        
    except Exception as e:
        print("LLM check failed:", e, flush=True)
        return False


