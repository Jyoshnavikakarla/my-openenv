import importlib

# check env
env_module = importlib.import_module("env.environment")
env_class = getattr(env_module, "EmailEnv")
print("ENV OK:", env_class)

# check grader
grader_module = importlib.import_module("graders.email_grader")
grader_class = getattr(grader_module, "EmailGrader")
print("GRADER OK:", grader_class)

# check tasks
for task in ["easy", "medium", "hard"]:
    module = importlib.import_module(f"tasks.{task}")
    cls = getattr(module, f"{task.capitalize()}Task")
    print(f"TASK {task.upper()} OK:", cls)