# minimal heuristic now
# later can be replaced by richer probabilistic logic

def prior_occurrence_check(retrieved_results):

    if len(retrieved_results) > 0:
        return True

    return False
