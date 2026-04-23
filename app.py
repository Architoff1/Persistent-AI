from flask import Flask, render_template, request, jsonify

from modules.memory_engine import (
    store_memory,
    retrieve_memory,
    restore_memory
)

from modules.salience import is_salient
if is_salient(user_msg):
    store_memory(user_msg)

from modules.prior_occurrence import (
    prior_occurrence_check
)

from modules.reconstruction_engine import (
    reconstruct_memory
)

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

    user_msg=request.json['message']

    results=retrieve_memory(user_msg)

    mode='direct recall'

    response=''

    if prior_occurrence_check(results):

        if len(results)>=1:
            response=(
             "I found related memory traces: "
             + results[0]
            )

        # if degraded memory case emerges
        if len(results)==1:
            mode='reconstruction'
            response=reconstruct_memory(
                user_msg,
                results
            )

    else:
        response='No prior event inferred.'

    # store ongoing conversation too
    store_memory(user_msg)

    return jsonify({
        'reply':response,
        'mode':mode
    })


if __name__=='__main__':
    app.run(debug=True)
