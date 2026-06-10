from transformers import pipeline
from transformers.utils import logging as hf_logging
from typing import Any, List
from pydantic import BaseModel, ConfigDict
from .Models import (MinimalSource,
                     MinimalSearchResults,
                     StudentSearchResults,
                     StudentSearchResultsAndAnswer,
                     UnansweredQuestion,
                     MinimalSourceOutput)
from pathlib import Path
import json
from tqdm import tqdm


class LLM(BaseModel):
    """Small wrapper around the Qwen text-generation pipeline that turns
    retrieved sources into a natural-language answer"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_name: str = "Qwen/Qwen3-0.6B"
    generator: Any = None

    def load_generator(self) -> Any:
        """Build the text-generation pipeline on first use and cache it on the
        instance, then return it."""
        if self.generator is None:
            hf_logging.set_verbosity_error()
            self.generator = pipeline(
                "text-generation", model=self.model_name)
        return self.generator

    def read_source_text(self, source: MinimalSource) -> str:
        """Read a source's text back from its file using its character offsets,
        used when the search results were saved without the chunk text."""
        try:
            with open(source.file_path, encoding="utf-8") as file:
                content = file.read()
        except OSError:
            return ""
        start = source.first_character_index
        end = source.last_character_index
        return content[start:end]

    def generate_answer(self, list_sources: MinimalSearchResults,
                        query: UnansweredQuestion,
                        max_tokens: int = 40) -> StudentSearchResultsAndAnswer:
        """Generate an answer for the query using the top retrieved source as
        context, and return it with the question and the source pointers"""
        output_sources: List[MinimalSourceOutput] = []

        if list_sources.retrieved_sources:
            top = list_sources.retrieved_sources[0]
            context = top.text or self.read_source_text(top)
        else:
            context = ""

        generator = self.load_generator()
        messages = [
            {"role": "system",
             "content": ("Answer the question using only the provided "
                         "context. Make a complete sentence to answer. If "
                         "the answer is not in the context, say 'I don't "
                         f"know'.\n\nContext:\n{context}")},
            {"role": "user", "content": query.question},
        ]

        prompt = generator.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        answer = generator(prompt, max_new_tokens=max_tokens,
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

    def handle_dataset(self, search_results_path: str,
                       save_directory: str) -> None:
        """Load a search-results file, generate an answer for each question it
        contains and save the answered dataset as JSON under save_directory"""
        try:
            with open(search_results_path, "r", encoding="utf-8") as file:
                json_data = file.read()
            obj = StudentSearchResults.model_validate_json(json_data)
        except Exception as e:
            print(f"Error: invalid input file: {e}")
            return

        list_search_result: List[MinimalSearchResults] = obj.search_results
        list_answer: List[StudentSearchResultsAndAnswer] = []

        for search_result in tqdm(list_search_result,
                                  desc="Answers generation"):

            query: UnansweredQuestion = UnansweredQuestion(
                question_id=search_result.question_id,
                question=search_result.question_str)

            answer_obj = self.generate_answer(search_result, query)
            list_answer.append(answer_obj)

        output = {
            "search_results": [a.model_dump() for a in list_answer],
            "k": obj.k,
        }

        save_path = Path(save_directory)
        file_name_dataset = Path(search_results_path).name

        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / file_name_dataset, "w",
                  encoding="utf-8") as output_file:
            json.dump(output, output_file, indent=4,
                      ensure_ascii=False)
        print("Saved student_search_results to "
              f"{save_path}/{file_name_dataset}")
