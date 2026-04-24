import faiss
import numpy as np
import pickle

from config import EMBED_DIM
from modules.affect_engine import infer_affect
from modules.embedding_client import get_embedding
from modules.salience import salience_score

index = faiss.IndexFlatL2(EMBED_DIM)

memory_records=[]


def store_memory(text):

    emb=np.array(
        [get_embedding(text)],
        dtype="float32"
    )

    index.add(emb)

    record={"text":text,
        "affect":infer_affect(text),
        "trace_pointers":[
            "semantic",
            "episodic"
            ]
            }
    memory_records.append(record)



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
                    memory_records[i]["text"]
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
