from .base_rephraser import BaseRephraser
from typing import Tuple

class ZeroShotRephraser(BaseRephraser):
    """Zero-shot email rephraser."""
    
    def __init__(self, llm_handler, prompt_template: str, generation_config: dict):
        super().__init__(llm_handler)
        self.prompt_template = prompt_template
        self.generation_config = generation_config
    
    def rephrase(self, subject: str, body: str) -> Tuple[str, str]:
        """Rephrase the email using zero-shot prompting."""
        prompt = self.prompt_template.format(subject=subject, body=body)
        output = self.llm_handler.generate(prompt, self.generation_config)
        return self.parse_output(output)