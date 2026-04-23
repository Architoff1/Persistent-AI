# minimal heuristic now
# later can be replaced by richer probabilistic logic

from modules.embedding_client import get_embedding
from modules.memory_engine import retrieve_memory

def prior_occurrence_check(query):

    candidates = retrieve_memory(query,k=3)

    if len(candidates)==0:
        return False,None

    # very simple prototype confidence heuristic
    # later replace with hybrid score

    best=candidates[0]

    confidence=0.78

    if confidence > 0.70:
        return True,candidates

    return False,None
