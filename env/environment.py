import random

from models.schemas import Observation, Reward
from graders.email_grader import EmailGrader


class EmailEnv:

    def __init__(self, task="easy"):
        self.task = task
        self.max_steps = 5
        self.grader = EmailGrader()
        random.seed(42)
        self.reset()

   
    def reset(self):
        self.step_count = 0
        self.current_email = self._get_email()
        return self._get_obs()

    
    def state(self):
        return self._get_obs()

    
    def step(self, action):
        self.step_count += 1

        # Get current observation
        obs = self._get_obs()

        # Convert Observation → dict for grader
        obs_dict = obs.dict()

        # Compute reward using external grader
        score = self.grader.grade(obs_dict, action)

        reward = Reward(score=score)

        # Check if episode done
        done = self.step_count >= self.max_steps

        # Move to next email
        self.current_email = self._get_email()

        return self._get_obs(), reward, done, {}

    
    def _get_obs(self):
        return Observation(
            email=self.current_email["text"],
            sender=self.current_email["sender"],
            step=self.step_count
        )

   
    def _get_email(self):
        emails = {
            "easy": [
                {"text": "I want a refund", "sender": "user1"},
                {"text": "App is crashing", "sender": "user2"},
                {"text": "How do I reset my password?", "sender": "user3"},
            ],
            "medium": [
                {"text": "Payment failed but money deducted", "sender": "user4"},
                {"text": "Cannot login to my account", "sender": "user5"},
                {"text": "Subscription not working after payment", "sender": "user6"},
            ],
            "hard": [
                {"text": "Charged twice and cannot login", "sender": "user7"},
                {"text": "App crashes and refund not processed", "sender": "user8"},
                {"text": "Payment deducted but account not activated and login fails", "sender": "user9"},
            ]
        }

        return random.choice(emails[self.task])