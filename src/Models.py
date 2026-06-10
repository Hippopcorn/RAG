from pydantic import BaseModel, Field
from typing import List
import uuid


class MinimalSource(BaseModel):
    """Represent a retrieved source as the raw text plus its location
    (file path and start/end character indices in the original file)."""
    text: str
    file_path: str
    first_character_index: int
    last_character_index: int


class MinimalSourceOutput(BaseModel):
    """Public-facing source description: only the file path and the start/end
    character indices, without the chunk text."""
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question waiting to be answered, paired with a unique identifier."""
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(BaseModel):
    """Ground-truth answer for a question, with the list of supporting sources."""
    sources: list[MinimalSource]
    answer: str


class MinimalSearchResults(BaseModel):
    """Retrieval output for a single question: the question itself and the
    list of sources returned by the retriever."""
    question_id: str
    question_str: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search results extended with the generated answer for the question."""
    answer: str


class RagDataset(BaseModel):
    """Collection of questions (answered or unanswered) used as a dataset."""
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class StudentSearchResults(BaseModel):
    """Aggregate retrieval results produced by a student run, with the ``k``
    value used at retrieval time."""
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Final RAG output for one question: the question, the retrieved sources
    (without text) and the generated answer."""
    question_id: str
    question: str
    retrieved_sources: List[MinimalSourceOutput]
    answer: str
