from transformers import pipeline
from typing import ClassVar, Any
from pydantic import BaseModel, ConfigDict


class LLM(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    generator: ClassVar[Any] = pipeline(
        "text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")

    def generate(self, prompt: str, max_tokens: int = 50):
        return self.generator(prompt, max_new_tokens=max_tokens)


if __name__ == "__main__":
    generator = LLM()
    response = generator.generate("L'intelligence artificielle permet de", 200)
    print(response)
