import os
from dotenv import load_dotenv
import openai

load_dotenv()

openai.api_key = os.getenv("OPEN_AI_API_KEY")

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.2,  # less random and more conscise
    )

    return response.choices[0].message["content"]

def get_reduced_data(prompt, model="gpt-3.5-turbo", max_points=5):
    completion = get_completion(prompt, model)
    # points = completion.split('\n')[:max_points]
    # return ' '.join(points)
    return completion