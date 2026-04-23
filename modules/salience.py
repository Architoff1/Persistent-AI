from modules.embedding_client import get_embedding
import numpy as np

def cosine(a,b):
    a=np.array(a); b=np.array(b)
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))


def salience_score(text,recent_embeddings,goal_embedding):

    e=get_embedding(text)

    novelty=1-max(
        cosine(e,r)
        for r in recent_embeddings
    )

    goal_rel=cosine(e,goal_embedding)

    info_density=min(
      len(text.split())/25,
      1.0
    )

    score=(
       0.4*novelty+
       0.4*goal_rel+
       0.2*info_density
    )

    return score
