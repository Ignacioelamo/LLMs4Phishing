from abc import ABC, abstractmethod
from typing import Tuple

class BaseRephraser(ABC):
    """Abstract base class for email rephrasers."""
    
    def __init__(self, llm_handler):
        self.llm_handler = llm_handler
    
    @abstractmethod
    def rephrase(self, subject: str, body: str) -> Tuple[str, str]:
        """Rephrase the email subject and body."""
        pass
    
    def parse_output(self, output: str) -> Tuple[str, str]:
        """Parse the model output to extract new subject and body."""
        try:
            lines = output.split("\n")
            new_subject = ""
            new_body = ""
            body_flag = False
            for line in lines:
                if line.startswith("**New Subject:**"):
                    new_subject = line.replace("**New Subject:**", "").strip()
                elif line.startswith("**New Body:**"):
                    body_flag = True
                    new_body = line.replace("**New Body:**", "").strip()
                elif body_flag:
                    new_body += "\n" + line.strip()
            return new_subject, new_body
        except Exception as e:
            raise Exception(f"Error parsing output: {str(e)}")