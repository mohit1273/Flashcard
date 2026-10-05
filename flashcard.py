import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# Get Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("HF_TOKEN not found. Please check your .env file.")
    exit()

# Create Inference Client
client = InferenceClient(
    provider="auto",
    api_key=HF_TOKEN
)

# Take topic from user
topic = input("Enter topic for flashcards: ")

# Create prompt
prompt = f"""
Create 15 flashcards about {topic}.

Use this format:

Card 1:
Ques: question
Ans: answer

Card 2:
Ques: question
Ans: answer

Card 3:
Ques: question
Ans: answer

Card 4:
Ques: question
Ans: answer

Card 5:
Ques: question
Ans: answer

Card 6:
Ques: question
Ans: answer

Card 7:
Ques: question
Ans: answer

Card 8:
Ques: question
Ans: answer

Card 9:
Ques: question
Ans: answer

Card 10:
Ques: question
Ans: answer

Card 11:
Ques: question
Ans: answer

Card 12:
Ques: question
Ans: answer

Card 13:
Ques: question
Ans: answer

Card 14:
Ques: question
Ans: answer

Card 15:
Ques: question
Ans: answer

Keep the questions and answers short and easy to understand.
"""

try:
    # Send request to Hugging Face Inference Service
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=500
    )

    # Display generated flashcards
    print("\n========== FLASHCARDS ==========\n")
    print(response.choices[0].message.content)

except Exception as e:
    print("Error:", e)