import os
from dotenv import load_dotenv, find_dotenv
from groq import Groq

load_dotenv(find_dotenv())
client=Groq(api_key=os.getenv("GROQ_API_KEY"))

def normal_chat(user_msg,recent_context=None):
    context_block=""
    if recent_context:
        context_block=f"""
Recent conversational context:
{recent_context}
"""


    prompt=f"""
You are Persistent AI,
a conversational assistant with persistent memory.

Respond naturally like a normal intelligent assistant.

Use recent context as memory evidence.

If the user asks whether something
was discussed before,
first check whether the supplied
context contains evidence of that,
and answer from it before assuming
blank slate.

{context_block}

User:
{user_msg}
"""


    chat=client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
             "role":"user",
             "content":prompt
            }
        ]
    )

    return (chat.choices[0].message.content)
