from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class ChatReq(BaseModel):
    user_id: str
    tradition: str
    text: str
    history: Optional[List[Dict[str, Any]]] = None

class ChatResp(BaseModel):
    answer: str
    quote: Optional[str] = None
    source: Optional[str] = None

class ProgressReq(BaseModel):
    user_id: str

class ProgressResp(BaseModel):
    xp: int
    streak: int
    state: str  # "sprout" | "branch" | "tree"

class MedStartReq(BaseModel):
    user_id: str
    tradition: str
    kind: str  # "breath" | "body_scan"

class MedStartResp(BaseModel):
    steps: List[str]

class MedCompleteReq(BaseModel):
    user_id: str

class LessonNextReq(BaseModel):
    user_id: str
    tradition: str

class Lesson(BaseModel):
    id: str
    title: str
    objective: str
    exercise: str

class LessonNextResp(BaseModel):
    done: bool = False
    lesson: Optional[Lesson] = None
    quote: Optional[str] = None
    source: Optional[str] = None

class LessonCompleteReq(BaseModel):
    user_id: str
    tradition: str
    lesson_id: str
