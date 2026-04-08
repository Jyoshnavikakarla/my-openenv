from graders.email_grader import EmailGrader

grader = EmailGrader()

# SAMPLE TEST
observation = {
    "email": "App is crashing",
    "sender": "user1"
}

action = {
    "category": "billing",
    "priority": "low",
    "response": "ok"
}

score = grader.grade(observation, action)

print("Score:", score)