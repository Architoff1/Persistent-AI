from modules.embedding_client import get_embedding
from config import SALIENCE_THRESHOLD
import numpy as np


def cosine(a,b):

    a=np.array(a)
    b=np.array(b)

    return np.dot(a,b)/(
        np.linalg.norm(a)*
        np.linalg.norm(b)
    )


def salience_score(
    text,
    recent_embeddings,
    goal_embedding
):

    e=get_embedding(text)


    # -------------------------
    # Semantic novelty
    # -------------------------

    if not recent_embeddings:

        novelty=1.0

    else:

        novelty=1-max(
            cosine(e,r)
            for r in recent_embeddings
        )


    # -------------------------
    # Goal relevance
    # -------------------------

    goal_rel=cosine(
        e,
        goal_embedding
    )


    # -------------------------
    # Information density
    # crude proxy for now
    # -------------------------

    info_density=min(
       len(text.split())/25,
       1.0
    )


    # -------------------------
    # Weighted salience score
    # -------------------------

    score=(
        0.4*novelty +
        0.4*goal_rel +
        0.2*info_density
    )


    return score



def should_store(
    text,
    recent_embeddings,
    goal_embedding
):

    return (
        salience_score(
            text,
            recent_embeddings,
            goal_embedding
        )
        > SALIENCE_THRESHOLD
    )
