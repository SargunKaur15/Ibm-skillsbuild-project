import gradio as gr
from transformers import pipeline, set_seed

# Load a simple, supported model from Hugging Face
chatbot = pipeline("text-generation", model="distilgpt2")
set_seed(42)

def get_response(user_message, mood):
    prompt = f"The user is feeling {mood}. They say: '{user_message}'\nRespond with kindness and support:\n"
    response = chatbot(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)[0]['generated_text']
    # Extract only the response part
    if prompt in response:
        return response.split(prompt)[-1].strip()
    return response.strip()


description = "Chat with a compassionate AI for emotional support, strategies, and empathy."

demo = gr.Interface(
    fn=get_response,
    inputs=[
        gr.Textbox(label="Your message", placeholder="e.g. I'm feeling overwhelmed with work."),
        gr.Dropdown(choices=["Good", "Okay", "Stressed", "Sad", "Angry", "Confused"], label="Mood")
    ],
    outputs=gr.Textbox(label="Bot Response"),
    title="🧠 AI Mental Health Chatbot (by code stormers)",
    description=description,
    theme="soft"
)

demo.launch()
