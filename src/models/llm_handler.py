from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Optional
import torch
from transformers import BitsAndBytesConfig

class LLMHandler:
    """Manages loading and inference for the language model."""
    
    def __init__(self, model_name: str, quantization_config: Optional[Dict[str, Any]] = None, device_map: str = "auto"):
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.device_map = device_map
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self):
        """Load the tokenizer and model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            quantization = None
            if self.quantization_config:
                quantization = BitsAndBytesConfig(**self.quantization_config)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization,
                device_map=self.device_map
            )
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def generate(self, prompt: str, generation_config: Dict[str, Any]) -> str:
        """Generate text based on the provided prompt."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer not loaded. Call load_model() first.")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=generation_config.get("max_new_tokens", 512),
            temperature=generation_config.get("temperature", 0.7),
            top_p=generation_config.get("top_p", 0.9),
            do_sample=generation_config.get("do_sample", True)
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)