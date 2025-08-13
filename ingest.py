import csv, os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from re import sub

load_dotenv()
API_KEY   = os.getenv("OPENAI_API_KEY")
STATE_DIR = os.getenv("STATE_DIR", os.path.join(os.getcwd(), "storage"))
CHROMA_DIR= os.getenv("CHROMA_DIR", os.path.join(STATE_DIR, "chroma"))

DB_PATH   = CHROMA_DIR
COLLECTION= "quotes_v1"

def canon_trad(s): 
    k = sub(r"\s+"," ", (s or "").strip().lower())
    return {"buddhism":"buddhism","buddhism (bodhi)":"buddhism","bodhi":"buddhism","zen":"zen","bonsai":"zen"}.get(k,k)

def load_quotes(csv_path=os.path.join(os.getenv("DATA_DIR", "data"), "quotes.csv")):
    docs, ids, metas = [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row["id"])
            docs.append(row["clean_quote"])
            metas.append({
                "tradition": canon_trad(row["tradition"]),
                "tags": row["tags"],
                "source_ref": row["source_ref"],
            })

    return ids, docs, metas

def main():
    ef = OpenAIEmbeddingFunction(api_key=API_KEY, model_name="text-embedding-3-small")
    client = chromadb.PersistentClient(path=DB_PATH)
    col = client.get_or_create_collection(name=COLLECTION, embedding_function=ef)
    ids, docs, metas = load_quotes()
    col.upsert(ids=ids, documents=docs, metadatas=metas)
    print(f"Ingested {len(ids)} quotes into {DB_PATH} / {COLLECTION}")

if __name__ == "__main__":
    main()
