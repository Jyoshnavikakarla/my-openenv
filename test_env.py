from graders.email_grader import EmailGrader

grader = EmailGrader()

# SAMPLE TEST
observation = {
    "email": "App is crashing",
    "sender": "user1"
}

action = {
    "category": "technical",
    "priority": "high",
    "response": "We will fix this crash issue soon."
}

score = grader.grade(observation, action)

print("Score:", score)