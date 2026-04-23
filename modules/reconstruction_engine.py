# current primitive reconstruction
# plug-point for Groq later

def reconstruct_memory(query,retrieved_results):

    if len(retrieved_results)==0:
        return None

    hypothesis=(
    "Probable reconstructed memory: "
    + retrieved_results[0]
    )

    return hypothesis


# future plug-point
# def groq_reconstruct(...):
#     pass
