import pandas as pd
from transformers import pipeline, AutoModel, AutoTokenizer, AutoConfig
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForMaskedLM
import torch

from abc import ABC, abstractmethod

class EmailRephraser(ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.generator = pipeline('text2text-generation', model=self.model_name, device=device)

    def rephrase_email(self, prompt, max_length=512):
        output = self.generator(prompt, max_length=max_length, num_return_sequences=1)
        return output[0]['generated_text']
    
