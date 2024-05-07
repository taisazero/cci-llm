# pip install openai
from openai import OpenAI

client = OpenAI(api_key = "OnuR-l5IlfYqF8HYoTOYHAcHOXCgL5xASQM5ooGHG6A")

# Set the OpenAI API base URL
client.base_url = "http://cci-llm.charlotte.edu/api/v1" #or machine IP: http://machine_ip:5000/v1

model_name = "Llama-3-70B"

# Define the conversation
conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who trained you?"},
]

# Send a chat completion request
response = client.chat.completions.create(
    model=model_name,
    messages=conversation,
    max_tokens=300,
    temperature =  0
)

reply = response.choices[0].message.content
print(reply)