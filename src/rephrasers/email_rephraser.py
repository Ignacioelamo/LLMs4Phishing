import pandas as pd
from transformers import pipeline, AutoModel, AutoTokenizer, AutoConfig
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForMaskedLM

from abc import ABC, abstractmethod

class EmailRephraser(ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)

        if self.config.model_type == 'gpt2':
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        elif self.config.model_type == 't5':
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        elif self.config.model_type == 'bert':
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        
        self.generator = pipeline('text-generation', model=self.model_name, tokenizer=self.tokenizer)

    def rephrase_email(self, prompt, max_length=100):
        output = self.generator(prompt, max_length=max_length, num_return_sequences=1)
        return output[0]['generated_text']
    
