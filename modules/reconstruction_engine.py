# current primitive reconstruction
# plug-point for Groq later
from groq import Groq

client=Groq(api_key=GROQ_API_KEY)

def reconstruct_memory(query,retrieved_results):

    context="\n".join(retrieved_results)

    prompt=f"""
Direct episodic memory is unavailable.

Using these residual traces:

{context}

Reconstruct the most probable prior memory the user is referring to.

State uncertainty if needed.
"""


    chat=client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
             "role":"user",
             "content":prompt
            }
        ]
    )

    return (
      chat.choices[0]
      .message.content
    )

# future plug-point
# def groq_reconstruct(...):
#     pass
