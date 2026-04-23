from flask import Flask, render_template, request, jsonify
from modules.embedding_client import get_embedding
from modules.salience import should_store
from modules.memory_engine import (store_memory,retrieve_memory)
from modules.prior_occurrence import (prior_occurrence_check)
from modules.reconstruction_engine import (reconstruct_memory)
from modules.memory_engine import save_memory

app = Flask(__name__)

# seed some memory
seeded=False

def seed_once():
    global seeded
    if seeded:
        return

    store_memory(
      "User proposed trace-based memory reconstruction."
    )

    store_memory(
      "User discussed context beyond time and identity."
    )

    store_memory(
      "User suggested reconstructing memories from residual traces."
    )

    seeded=True


@app.route('/')
def home():
    seed_once()
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():

    user_msg = request.json['message']


    # --------------------------------
    # Prior occurrence inference
    # --------------------------------

    prior_event, results = prior_occurrence_check(user_msg)
    mode='novel interaction'
    if prior_event:
        mode='direct recall'

        response=(
          "I remember you previously mentioned: "
          + results[0]
        )


        # Important Upgrade for later stage degraded-memory style case, its a demo we will later Not ideal. Eventually should use prior confidence + damage signal. But for demo, acceptable.
        if len(results)==1:

            mode='reconstruction'

            response=reconstruct_memory(
                user_msg,
                results
            )


    else:

        response=(
         "This seems like a new interaction. "
         "Tell me more."
        )


    # --------------------------------
    # Salience-gated storage
    # --------------------------------

    # placeholder recent embeddings for now
    recent_embeddings = [
        get_embedding(r)
        for r in results
    ] if results else []

    goal_embedding=get_embedding(
       "persistent ai research"
    )


    if should_store(user_msg,recent_embeddings,goal_embedding):
        store_memory(user_msg)
        save_memory()



    return jsonify({
        'reply':response,
        'mode':mode
    })

if __name__=='__main__':
    app.run(debug=True)
