from pydantic import BaseModel, Field
from typing import List
import uuid


class MinimalSource(BaseModel):
    """ Instanciation of a source, store the path and the
        indexs of the start and the end of the source """
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """ Instanciation of a question and its ID """
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(BaseModel):
    """ Store the answer and the sources """
    sources: list[MinimalSource]
    answer: str


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class RagDataset(BaseModel):
    """ Store all the questions and answer instances """
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    search_results: List[MinimalAnswer]
