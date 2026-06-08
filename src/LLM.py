from transformers import pipeline
from typing import ClassVar, Any, List
from pydantic import BaseModel, ConfigDict
from .Models import (MinimalSearchResults,
                     StudentSearchResultsAndAnswer,
                     UnansweredQuestion,
                     MinimalSourceOutput)


class LLM(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    generator: ClassVar[Any] = pipeline(
        "text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")

    def generate_answer(self, list_sources: MinimalSearchResults,
                        query: UnansweredQuestion,
                        max_tokens: int = 50) -> StudentSearchResultsAndAnswer:
        retrieved_sources: str = ""
        output_sources: List[MinimalSourceOutput] = []
        for source in list_sources.retrieved_sources:
            retrieved_sources += source.text

        prompt: str = (
            "<|im_start|>system\n"
            "Answer the question using only the provided context. "
            "Make a complete sentence to answer. "
            "If the answer is not in the context, say 'I don't know'.\n\n"
            f"Context:\n{retrieved_sources}<|im_end|>\n"
            f"<|im_start|>user\n{query.question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        answer = self.generator(prompt, max_new_tokens=max_tokens,
                                return_full_text=False)
        answer_text = answer[0]['generated_text']
        if answer_text:
            answer_text = answer_text[0].upper() + answer_text[1:]

        for source in list_sources.retrieved_sources:
            output_source = MinimalSourceOutput(
                file_path=source.file_path,
                first_character_index=source.first_character_index,
                last_character_index=source.last_character_index
                )
            output_sources.append(output_source)

        answer_object = StudentSearchResultsAndAnswer(
            question_id=query.question_id,
            question=query.question,
            retrieved_sources=output_sources,
            answer=answer_text
        )
        return answer_object
