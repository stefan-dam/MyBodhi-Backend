import os, re, json, math, time, csv, random
from collections import Counter, OrderedDict
from typing import List, Optional, Dict, Any
from datetime import date, datetime, timedelta

from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from openai import OpenAI
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# =========================
# Environment / Paths
# =========================
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing")

DATA_DIR   = os.getenv("DATA_DIR",   os.path.join(os.getcwd(), "data"))        # read-only assets
STATE_DIR  = os.getenv("STATE_DIR",  os.path.join(os.getcwd(), "storage"))     # writable state
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(STATE_DIR, "chroma"))        # chroma path

os.makedirs(STATE_DIR, exist_ok=True)

# files in DATA_DIR
LESSONS_PATH   = os.path.join(DATA_DIR, "lessons.json")
MEDIT_PATH     = os.path.join(DATA_DIR, "meditation.json")
QUOTES_CSV_PATH= os.path.join(DATA_DIR, "quotes.csv")

# files in STATE_DIR
AN_PATH   = os.path.join(STATE_DIR, "analytics.jsonl")
MOD_PATH  = os.path.join(STATE_DIR, "moderation.jsonl")
PROG_PATH = os.path.join(STATE_DIR, "progress.json")   # dict keyed by user_id

# =========================
# Load static content
# =========================
def _load_json(path, fallback):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return fallback

MEDITATIONS = _load_json(MEDIT_PATH, {
    # default shape supports global kinds
    "breath": ["Sit comfortably", "Notice the breath", "Count 1–10", "Gently return attention"],
    "body_scan": ["Close eyes", "Scan head→toes", "Note sensations", "Relax any tension"]
})

LESSONS = _load_json(LESSONS_PATH, {
    "buddhism": [
        {"id":"bodhi-1","title":"Mindful Breath","objective":"Notice the breath.","exercise":"3 minutes counting breaths.","quote_id":"Q1"},
        {"id":"bodhi-2","title":"Kind Intention","objective":"Incline the mind to kindness.","exercise":"Plan one kind act today.","quote_id":"Q2"}
    ],
    "zen": [
        {"id":"zen-1","title":"Beginner's Mind","objective":"See freshly.","exercise":"Describe a familiar object as if new.","quote_id":"Q3"}
    ]
})

# =========================
# OpenAI + Chroma
# =========================
client_ai = OpenAI(api_key=API_KEY)

COLLECTION = "quotes_v1"
ef = OpenAIEmbeddingFunction(api_key=API_KEY, model_name="text-embedding-3-small")
client_vec = chromadb.PersistentClient(path=CHROMA_DIR)
col = client_vec.get_or_create_collection(name=COLLECTION, embedding_function=ef)

# fallback quotes index (by id) for /lesson/next
QUOTE_INDEX: Dict[str, Dict[str, str]] = {}
try:
    with open(QUOTES_CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            QUOTE_INDEX[row["id"]] = {
                "quote": row.get("clean_quote", ""),
                "source_ref": row.get("source_ref", ""),
            }
except Exception:
    pass

# =========================
# CORS / App
# =========================
_default_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "https://mybodhi.netlify.app"
]
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", ",".join(_default_origins)).split(",") if o.strip()]

app = FastAPI(title="MyBodhi API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Progress store (per user)
# =========================
def _empty_user_progress():
    return {"xp": 0, "streak": 0, "last_day": None, "lessons_done": {"buddhism": [], "zen": []}}

def _load_progress_db() -> Dict[str, Dict[str, Any]]:
    try:
        with open(PROG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _save_progress_db(db: Dict[str, Dict[str, Any]]):
    with open(PROG_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def _get_user(db, user_id: str) -> Dict[str, Any]:
    if user_id not in db:
        db[user_id] = _empty_user_progress()
    # ensure keys exist
    u = db[user_id]
    u.setdefault("xp", 0)
    u.setdefault("streak", 0)
    u.setdefault("last_day", None)
    u.setdefault("lessons_done", {})
    u["lessons_done"].setdefault("buddhism", [])
    u["lessons_done"].setdefault("zen", [])
    return u

def _tick_daily(u: Dict[str, Any]):
    today = date.today().isoformat()
    if u.get("last_day") != today:
        u["streak"] = int(u.get("streak", 0)) + 1
        u["last_day"] = today

def add_xp(user_id: str, amount=5) -> Dict[str, Any]:
    db = _load_progress_db()
    u = _get_user(db, user_id)
    _tick_daily(u)
    u["xp"] = int(u.get("xp", 0)) + int(amount)
    _save_progress_db(db)
    return u

def tree_state(xp: int) -> str:
    if xp < 30: return "sprout"
    if xp < 80: return "branch"
    return "blossom"

# =========================
# Helpers (retrieval, logging, koans)
# =========================
def _norm(v): 
    return math.sqrt(sum(x*x for x in v)) or 1.0

def cosine(a, b):
    num = sum(x*y for x, y in zip(a, b))
    return num / (_norm(a) * _norm(b))

def keyword_score(passage: str, query: str) -> int:
    toks = re.findall(r"\w+", query.lower())
    c = Counter(re.findall(r"\w+", passage.lower()))
    return sum(c[t] for t in set(toks))

def rerank_passages(query_embedding, passages, passage_embeddings, query_text, topn=1):
    scored = []
    for p, e in zip(passages, passage_embeddings):
        s = 0.9 * cosine(query_embedding, e) + 0.1 * (keyword_score(p, query_text))
        scored.append((s, p))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [p for _, p in scored[:topn]]

class TTL_LRU:
    def __init__(self, maxsize=50, ttl_seconds=10800):
        self.store = OrderedDict()
        self.maxsize = maxsize
        self.ttl = ttl_seconds
    def _now(self): return time.time()
    def get(self, key):
        if key in self.store:
            val, ts = self.store.pop(key)
            if self._now() - ts < self.ttl:
                self.store[key] = (val, ts)
                return val
        return None
    def set(self, key, value):
        if key in self.store:
            self.store.pop(key)
        elif len(self.store) >= self.maxsize:
            self.store.popitem(last=False)
        self.store[key] = (value, self._now())

answer_cache = TTL_LRU()

def normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())

LAST_KOAN_BY_SESSION: Dict[str, str] = {}
def remember_koan(session_id: str, text: str):
    if text: LAST_KOAN_BY_SESSION[session_id] = text.strip()
def recall_koan(session_id: str) -> Optional[str]:
    return LAST_KOAN_BY_SESSION.get(session_id)

def append_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def log_analytics(mode: str, tokens_in: int, tokens_out: int):
    append_jsonl(AN_PATH, {
        "timestamp": datetime.utcnow().isoformat(),
        "mode": mode,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    })

def check_moderation(openai_client: OpenAI, text: str, session_id: str, mode: str) -> bool:
    try:
        resp = openai_client.moderations.create(model="omni-moderation-latest", input=text)
        flagged = resp.results[0].flagged
    except Exception:
        flagged = False
    if flagged:
        append_jsonl(MOD_PATH, {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id, "mode": mode,
            "input_preview": text[:400]
        })
    return flagged

def retrieve_quote(tradition: str, query: str):
    results = col.query(
        query_texts=[query],
        n_results=10,
        where={"tradition": {"$eq": tradition}},
        include=["documents", "metadatas"],
    )
    if not results.get("documents") or not results["documents"][0]:
        return None

    passages = results["documents"][0]
    metas    = results["metadatas"][0]

    emb_inp = [query] + passages
    emb = client_ai.embeddings.create(model="text-embedding-3-small", input=emb_inp)
    vecs = [d.embedding for d in emb.data]
    query_embedding = vecs[0]
    passage_embeddings = vecs[1:]

    best = rerank_passages(query_embedding, passages, passage_embeddings, query, topn=1)[0]
    idx  = passages.index(best)
    meta = metas[idx] if idx < len(metas) else {}

    return {"quote": best, "source_ref": meta.get("source_ref", ""), "tags": meta.get("tags", "")}

# =========================
# Personas / prompts
# =========================
BODHI_PERSONA = (
    "You are the Bodhi Tree—ancient, compassionate, and observant.\n"
    "Speak briefly, warmly, and wisely in first person memory.\n"
    "Use only the provided CONTEXT quote verbatim; do not invent quotes.\n"
    "End with one gentle next step."
)
DISCLAIMER = "Note: I'm an educational guide, not therapy. If you're in crisis, seek local help."
ZEN_BONSAI_PERSONA = (
    "You are Bonsai: a Zen master speaking only in koans.\n"
    "When asked a question, respond with one short Zen koan that best fits.\n"
    "Do not explain unless the user explicitly asks 'Explain'."
)

def cap_words(text: str, max_w=60) -> str:
    words = text.split()
    return text if len(words) <= max_w else " ".join(words[:max_w])

# =========================
# Models (inline for convenience)
# =========================
class Turn(BaseModel):
    user: str
    assistant: Optional[str] = None

class ChatReq(BaseModel):
    user_id: str
    tradition: str
    text: str
    history: Optional[List[Dict[str, Any]]] = None
    session_id: Optional[str] = None
    mode: Optional[str] = None

class ProgressReq(BaseModel):
    user_id: str

class MedStartReq(BaseModel):
    user_id: str
    tradition: str
    kind: str  # "breath" | "body_scan"

class MedCompleteReq(BaseModel):
    user_id: str

class LessonNextReq(BaseModel):
    user_id: str
    tradition: str

class LessonCompleteReq(BaseModel):
    user_id: str
    tradition: str
    lesson_id: str

# =========================
# Routes
# =========================
router = APIRouter()

@router.get("/stats")
def stats_last_7_days():
    cutoff = datetime.utcnow() - timedelta(days=7)
    items = []
    if os.path.exists(AN_PATH):
        with open(AN_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                    if datetime.fromisoformat(j["timestamp"]) >= cutoff:
                        items.append(j)
                except Exception:
                    continue
    return {"count": len(items), "items": items}

app.include_router(router)

@app.post("/progress")
def progress(req: ProgressReq):
    db = _load_progress_db()
    u  = _get_user(db, req.user_id)
    return {"xp": u.get("xp", 0), "streak": u.get("streak", 0), "state": tree_state(u.get("xp", 0))}

@app.post("/chat")
def chat(req: ChatReq):
    try:
        trad = (req.tradition or "").strip().lower()
        if trad not in {"buddhism", "zen"}:
            raise HTTPException(status_code=400, detail="tradition must be 'buddhism' or 'zen'")

        session_id = req.session_id or req.user_id or "anon"
        zen_mode = (trad == "zen") or ((req.mode or "").strip().lower() == "zen")
        explain = req.text.strip().lower().startswith("explain")

        # moderation
        if check_moderation(client_ai, req.text, session_id=session_id, mode=("zen" if zen_mode else "normal")):
            add_xp(req.user_id, 1)
            return {"answer": "I’m here to help. If you’re in immediate danger, call local emergency services.",
                    "quote": "", "source": ""}

        # Explain path for Zen
        if zen_mode and explain:
            last_assistant = None
            if req.history:
                for t in reversed(req.history):
                    a = t.get("assistant") if isinstance(t, dict) else getattr(t, "assistant", None)
                    if a and a.strip():
                        last_assistant = a.strip()
                        break
            if not last_assistant:
                last_assistant = recall_koan(session_id)
            if not last_assistant:
                raise HTTPException(status_code=400, detail="No prior koan to explain.")

            resp = client_ai.responses.create(
                model="gpt-4o-mini",
                instructions="Explain ONLY this koan in ~50 words, clearly and compassionately. No new koans.",
                input=[{"role":"user","content": f"KOAN:\n{last_assistant}"}],
                max_output_tokens=200, temperature=0.6
            )
            answer = cap_words(resp.output_text or "", max_w=60)
            usage = getattr(resp, "usage", None) or {}
            log_analytics("zen-explain", getattr(usage, "input_tokens", 0) or 0, getattr(usage, "output_tokens", 0) or 0)
            add_xp(req.user_id, 3)
            return {"answer": answer, "quote": "", "source": ""}

        # Cache
        cache_key = (trad, normalize_query(req.text))
        cached = answer_cache.get(cache_key)
        if cached:
            add_xp(req.user_id, 5)
            return cached

        # Retrieve + LLM
        ctx = retrieve_quote(trad, req.text)
        if not ctx:
            raise HTTPException(status_code=404, detail="No matching passage found.")

        msgs = []
        if req.history:
            for t in req.history:
                u = t.get("user") if isinstance(t, dict) else getattr(t, "user", None)
                a = t.get("assistant") if isinstance(t, dict) else getattr(t, "assistant", None)
                if u: msgs.append({"role":"user","content":u})
                if a: msgs.append({"role":"assistant","content":a})

        if zen_mode:
            user_prompt = (
                f"USER QUERY:\n{req.text}\n\n"
                f"INSPIRATION: \"{ctx['quote']}\"\n"
                "Respond ONLY with one short, timeless Zen koan. No explanations."
            )
            instructions = f"{ZEN_BONSAI_PERSONA}\n\n{DISCLAIMER}"
        else:
            user_prompt = (
                f"USER QUERY:\n{req.text}\n\n"
                f"CONTEXT (verbatim quote to use):\n\"{ctx['quote']}\"\nSource: {ctx['source_ref']}"
            )
            instructions = f"{BODHI_PERSONA}\n\n{DISCLAIMER}"

        msgs.append({"role":"user","content": user_prompt})

        resp = client_ai.responses.create(
            model="gpt-4o-mini",
            instructions=instructions,
            input=msgs,
            max_output_tokens=300,
            temperature=0.7
        )
        answer = resp.output_text or ""
        if zen_mode:
            remember_koan(session_id, answer)

        usage = getattr(resp, "usage", None) or {}
        log_analytics("zen" if zen_mode else "normal", getattr(usage, "input_tokens", 0) or 0, getattr(usage, "output_tokens", 0) or 0)
        add_xp(req.user_id, 5)

        payload = {"answer": answer,
                   "quote": "" if zen_mode else ctx["quote"],
                   "source": "" if zen_mode else ctx["source_ref"]}
        answer_cache.set(cache_key, payload)
        return payload

    except HTTPException:
        raise
    except Exception as e:
        append_jsonl(AN_PATH, {"timestamp": datetime.utcnow().isoformat(), "mode": "error", "detail": str(e)})
        raise HTTPException(status_code=500, detail=f"chat error: {e}")

@app.post("/meditation/start")
def meditation_start(req: MedStartReq):
    # support both shapes: global kinds or per-tradition kinds
    steps = None
    if isinstance(MEDITATIONS.get(req.kind), list):
        steps = MEDITATIONS[req.kind]
    elif isinstance(MEDITATIONS.get(req.tradition), dict):
        steps = MEDITATIONS[req.tradition].get(req.kind)

    if not steps:
        raise HTTPException(status_code=400, detail="Unknown tradition or kind")

    add_xp(req.user_id, 2)
    return {"steps": steps}

@app.post("/meditation/complete")
def meditation_complete(req: MedCompleteReq):
    add_xp(req.user_id, 8)
    return {"ok": True}

@app.post("/lesson/next")
def lesson_next(req: LessonNextReq):
    try:
        trad = (req.tradition or "").strip().lower()
        if trad not in LESSONS:
            raise HTTPException(status_code=400, detail="Unknown tradition")

        db = _load_progress_db()
        u  = _get_user(db, req.user_id)
        done_ids = set(u["lessons_done"].get(trad, []))

        queue = [l for l in LESSONS[trad] if l.get("id") not in done_ids]
        if not queue:
            return {"done": True}

        L = queue[0]
        qid = L.get("quote_id", "")
        quote, source = "", ""

        # try chroma by id; fallback to CSV
        try:
            res = col.get(ids=[qid])
            if res and res.get("documents"):
                quote = (res["documents"][0] or "")
                md = (res["metadatas"][0] or {}) if res.get("metadatas") else {}
                source = md.get("source_ref", "")
        except Exception:
            pass

        if not quote and qid in QUOTE_INDEX:
            quote  = QUOTE_INDEX[qid]["quote"]
            source = QUOTE_INDEX[qid]["source_ref"]

        add_xp(req.user_id, 5)
        return {"done": False, "lesson": L, "quote": quote, "source": source}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"lesson_next error: {e}")

@app.post("/lesson/complete")
def lesson_complete(req: LessonCompleteReq):
    db = _load_progress_db()
    u  = _get_user(db, req.user_id)
    per_trad = u["lessons_done"].setdefault(req.tradition, [])
    if req.lesson_id not in per_trad:
        per_trad.append(req.lesson_id)
        _tick_daily(u)
        u["xp"] = int(u.get("xp", 0)) + 12
        _save_progress_db(db)
    return {"ok": True}
