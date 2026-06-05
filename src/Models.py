from pydantic import BaseModel, Field
from typing import List
import uuid


class MinimalSource(BaseModel):
    """ Represent a source, and store the path and the
        indexs of the start and the end """
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """ Represent a question and its ID """
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(BaseModel):
    """ Represent the answer and the sources """
    sources: list[MinimalSource]
    answer: str


class MinimalSearchResults(BaseModel):
    """ Represent the question and the search results """
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """ Represent the answer """
    answer: str


class RagDataset(BaseModel):
    """ Store all the questions and answer instances """
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class StudentSearchResults(BaseModel):
    """ Represent the search results """
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    """ Represent the search results and the answer """
    search_results: List[MinimalAnswer]
