from env.environment import EmailEnv
# app.py
import gradio as gr
from inference import process_email

def run_email(email_text):
    result = process_email(email_text)
    return f"""
Category: {result['category']}
Priority: {result['priority']}
Response: {result['response']}
Reward Points: {result['reward']}
Resolved: {result['solved']}
Emails Received: {result['emails_received']}
Emails Sent: {result['emails_sent']}
"""

demo = gr.Interface(
    fn=run_email,
    inputs=gr.Textbox(lines=5, placeholder="Type your email here..."),
    outputs="text",
    title="Email Agent",
    description="Type an email and get automated response, reward points, and status."
)

demo.launch()
env = EmailEnv()

def reset():
    return env.reset()

def step(action):
    return env.step(action)