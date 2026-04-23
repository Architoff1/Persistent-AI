import faiss
import numpy as np
import pickle

from config import EMBED_DIM
from modules.embedding_client import get_embedding


index = faiss.IndexFlatL2(EMBED_DIM)

memory_records=[]


def store_memory(text):

    emb=np.array(
        [get_embedding(text)],
        dtype="float32"
    )

    index.add(emb)

    memory_records.append(text)



def retrieve_memory(query,k=3):

    q=np.array(
        [get_embedding(query)],
        dtype="float32"
    )

    D,I=index.search(q,k)

    results=[]

    for i in I[0]:

        if i!=-1:

            if memory_records[i] != "[DELETED]":
                results.append(
                    memory_records[i]
                )

    return results



def delete_memory(idx):

    global memory_records

    memory_records[idx]="[DELETED]"

    print("Memory deleted.")


def restore_memory(text):

    print("Reconsolidating memory...")

    store_memory(text)



def save_memory():

    faiss.write_index(
        index,
        "memory/faiss_index.bin"
    )

    with open(
        "memory/memory_records.pkl",
        "wb"
    ) as f:

        pickle.dump(
            memory_records,
            f
        )
