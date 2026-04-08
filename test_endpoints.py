import requests

task = "easy"
obs = requests.get(f"http://127.0.0.1:8080/reset/{task}").json()
print("Observation:", obs)

action = {
    "category": "technical",
    "priority": "high",
    "response": "We will fix this issue."
}

reward = requests.post(f"http://127.0.0.1:8080/step/{task}", json=action).json()
print("Reward:", reward)