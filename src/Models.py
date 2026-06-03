from pydantic import BaseModel, Field
import json


class Source(BaseModel):
    """ Instanciation of a source, store the path and the
        indexs of the start and the end of the source """
    file_path: str
    firt_index: int
    last_index: int


class Question(BaseModel):
    """ Instanciation of a question and its ID """
    question_id: str = Field(min_length=5)
    text: str


class Answer(BaseModel):
    """ Store the answer and the sources """
    question: Question
    answer: str
    sources: list[Source] = []


class RagDataset(BaseModel):
    """ Store all the questions and answer instances """
    questions_list: list[Question] = []
    answer_list: list[Answer] = []
    
