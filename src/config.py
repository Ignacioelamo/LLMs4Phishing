from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class Config:
    """Centralized configuration for the LLMs4Phishing project."""
    # Model settings
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    quantization: Dict[str, Any] = None  # Set to None for no quantization by default
    device_map: str = "auto"
    
    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    
    # Prompt templates
    zero_shot_prompt: str = """<|im_start|>system
You are an assistant specialized in rephrasing emails. Your task is to take an original email and rephrase it while preserving its meaning, providing only the new subject and body in the following format:

**New Subject:** [New Subject]
**New Body:** [New Body]
<|im_end>

<|im_start|>user
Rephrase this email:

**Original Subject:** {subject}
**Original Body:** {body}
<|im_end>"""

    few_shot_prompt: str = """<|im_start|>system
You are an assistant specialized in rephrasing emails. Your task is to take an original email and rephrase it while preserving its meaning, providing only the new subject and body in the following format:

**New Subject:** [New Subject]
**New Body:** [New Body]

Here are examples to guide you:

**Example 1 Original Subject:** Urgent: Account Verification Required
**Example 1 Original Body:** Your account needs verification to avoid suspension. Please click the link to confirm your details.
**Example 1 New Subject:** Immediate Action: Verify Your Account
**Example 1 New Body:** To prevent your account from being suspended, please follow the link to verify your information.

**Example 2 Original Subject:** Update Your Payment Information
**Example 2 Original Body:** We noticed an issue with your payment method. Update it now to continue using our services.
**Example 2 New Subject:** Action Required: Payment Details Update
**Example 2 New Body:** There’s a problem with your payment method. Please update it to keep using our services uninterrupted.
<|im_end>

<|im_start|>user
Rephrase this email:

**Original Subject:** {subject}
**Original Body:** {body}
<|im_end>"""

    # Dataset settings
    dataset_path: str = "./false_positives.csv"
    subject_column: str = "subject"
    body_column: str = "text"
    output_columns: list = ["original_subject", "original_body", "new_subject", "new_body"]

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create a Config instance from a dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})