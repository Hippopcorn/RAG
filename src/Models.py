from pydantic import BaseModel, Field
from typing import List
import uuid


class MinimalSource(BaseModel):
    """A retrieved chunk: its text plus its file path and start/end positions
    in the original file. ``text`` is optional so search results saved without
    it (lighter output) can still be reloaded."""
    text: str = ""
    file_path: str
    first_character_index: int
    last_character_index: int


class MinimalSourceOutput(BaseModel):
    """Public source info: file path and start/end positions, without the
    chunk text"""
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question with a unique id, waiting to be answered"""
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(BaseModel):
    """A question's correct answer with the list of supporting sources"""
    sources: list[MinimalSource]
    answer: str


class MinimalSearchResults(BaseModel):
    """Retrieval result for one question: the question and the sources
    returned by the retriever"""
    question_id: str
    question_str: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search results plus the generated answer for the question"""
    answer: str


class RagDataset(BaseModel):
    """A set of questions (answered or not) used as a dataset"""
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class StudentSearchResults(BaseModel):
    """All retrieval results from one run, with the k value used"""
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Final output for one question: the question, the retrieved sources
    (without text) and the generated answer"""
    question_id: str
    question: str
    retrieved_sources: List[MinimalSourceOutput]
    answer: str
