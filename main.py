from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the embeddings
df = pd.read_pickle('tafseer_embeddings.pkl')

app = FastAPI()

class SearchRequest(BaseModel):
    sentence: str

class SearchResult(BaseModel):
    surah: int
    verse: int
    similarity: float

def search_tafseer(search_sentence: str, similarity_threshold: float = 0.2) -> List[Dict[str, int | float]]:
    search_embedding = model.encode(search_sentence)

    df['similarity'] = df['embedding'].apply(lambda x: util.cos_sim(x, search_embedding).item())

    # Filter results based on the similarity threshold
    filtered_df = df[df['similarity'] >= similarity_threshold]

    # Sort the filtered DataFrame by similarity in descending order
    sorted_df = filtered_df.sort_values(by='similarity', ascending=False)

    results = []
    for _, row in sorted_df.iterrows():
        results.append({
            "surah": int(row['sorah']),
            "verse": int(row['ayah']) + 1,
            "similarity": float(row['similarity'])
        })

    return results

@app.post("/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    results = search_tafseer(request.sentence, similarity_threshold=0.2) # Threshold is now a parameter
    return results