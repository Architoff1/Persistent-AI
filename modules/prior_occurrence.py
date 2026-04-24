import numpy as np
from modules.affect_engine import infer_affect
import json
from modules.embedding_client import (get_embedding)
from modules.memory_engine import (retrieve_memory)


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
        k=5
    )


    if len(candidates)==0:
        return False,[]


    query_emb = get_embedding(query)
    query_affect=json.loads(infer_affect(query))


    # ------------------------
    # evidence from retrieved traces
    # ------------------------

    similarities=[]

    for c in candidates:
        c_emb=get_embedding(c)
        sim=cosine(query_emb,c_emb)
        cand_affect=json.loads(infer_affect(c))
        
        emotion_overlap=(
            query_affect["fear"]*cand_affect["fear"]+
            query_affect["joy"]*cand_affect["joy"]+
            query_affect["sadness"]*cand_affect["sadness"]+
            query_affect["surprise"]*cand_affect["surprise"]+
            query_affect["urgency"]*cand_affect["urgency"]
        )
        
        combined_score=(0.7*sim +0.3*emotion_overlap)
        similarities.append(combined_score)


    # use strongest match as confidence
    confidence=(0.7*max(similarities)+0.3*np.mean(similarities))
    dynamic_threshold=(np.mean(similarities)+0.3*np.std(similarities))
    
    if (confidence > dynamic_threshold and confidence > 0.76):
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
