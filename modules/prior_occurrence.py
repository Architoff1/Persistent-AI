import numpy as np

from config import PRIOR_EVENT_THRESHOLD

from modules.embedding_client import (
    get_embedding
)

from modules.memory_engine import (
    retrieve_memory
)



def cosine(a,b):

    a=np.array(a)
    b=np.array(b)

    return np.dot(a,b)/(
        np.linalg.norm(a)*
        np.linalg.norm(b)
    )



def prior_occurrence_check(query):

    candidates = retrieve_memory(
        query,
        k=3
    )


    if len(candidates)==0:
        return False,[]


    query_emb = get_embedding(query)


    # ------------------------
    # evidence from retrieved traces
    # ------------------------

    similarities=[]

    for c in candidates:

        c_emb=get_embedding(c)

        sim=cosine(
            query_emb,
            c_emb
        )

        similarities.append(sim)


    # use strongest match as confidence
    confidence=max(similarities)


    # optional softer aggregate alternative:
    # confidence=np.mean(similarities)


    if confidence > PRIOR_EVENT_THRESHOLD:

        return True,candidates


    return False,[]
    
'''
slight improvement - 
Instead of:
confidence=max(similarities)

maybe:
confidence = weighted combination of
max similarity
+ mean similarity

because single maxima can be noisy.
_________________________________________________________________

we are estimating P(prior event | evidence) by using Cosine similarity:
cos(θ)=
∥a∥∥b∥
a⋅b

This is vector geometry.
_________________________________________________________________

multi-qa-mpnet-base-dot-v1

means the model’s learned semantic manifold is doing part of the reasoning.

We might use later stages:

1- Bayesian inference over memory hypotheses
2- uncertainty calibration
3- learned salience predictors
4- temporal memory graphs
5- surprise-based predictive coding signals

We are not fully there.
_________________________________________________________________
We are using:
1- embeddings
2- vector similarity
3- probabilistic thresholds
4- weighted scoring
5- generative reconstruction

'''
