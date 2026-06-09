from transformers import pipeline
from typing import ClassVar, Any, List
from pydantic import BaseModel, ConfigDict
from .Models import (MinimalSearchResults,
                     StudentSearchResults,
                     StudentSearchResultsAndAnswer,
                     UnansweredQuestion,
                     MinimalSourceOutput)
from pathlib import Path
import json
from tqdm import tqdm


class LLM(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    generator: ClassVar[Any] = pipeline(
        "text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")

    def generate_answer(self, list_sources: MinimalSearchResults,
                        query: UnansweredQuestion,
                        max_tokens: int = 50) -> StudentSearchResultsAndAnswer:
        output_sources: List[MinimalSourceOutput] = []

        prompt: str = (
            "<|im_start|>system\n"
            "Answer the question using only the provided context. "
            "Make a complete sentence to answer. "
            "If the answer is not in the context, say 'I don't know'.\n\n"
            f"Context:\n{list_sources.retrieved_sources[0].text}<|im_end|>\n"
            f"<|im_start|>user\n{query.question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        answer = self.generator(prompt, max_new_tokens=max_tokens,
                                max_length=None,
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

    def handle_dataset(self, search_results_path: str, save_directory: str):
        try:
            with open(search_results_path, "r", encoding="utf-8") as file:
                json_data = file.read()
            obj = StudentSearchResults.model_validate_json(json_data)
        except Exception as e:
            print(e)
            exit()

        list_search_result: List[MinimalSearchResults] = obj.search_results
        list_answer: List[StudentSearchResultsAndAnswer] = []

        for search_result in tqdm(list_search_result,
                                  desc="Answers generation"):

            query: UnansweredQuestion = UnansweredQuestion(
                question_id=search_result.question_id,
                question=search_result.question_str)

            answer_obj = self.generate_answer(search_result, query)
            list_answer.append(answer_obj)
        json_ready_list = [answer.model_dump() for answer in list_answer]

        save_path = Path(save_directory)
        file_name_dataset = Path(search_results_path).name

        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / file_name_dataset, "w",
                  encoding="utf-8") as output_file:
            json.dump(json_ready_list, output_file, indent=4,
                      ensure_ascii=False)
        print("Saved student_search_results to "
              f"{save_path}/{file_name_dataset}")
