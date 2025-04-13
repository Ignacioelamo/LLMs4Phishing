from transformers import AutoTokenizer, AutoModelForCausalLM
from bitsandbytes import BitsAndBytesConfig
import torch

class EmailRephraser:
    """
    A class to handle email rephrasing using a specified language model with optional quantization.
    Supports zero-shot and few-shot prompting.
    """
    
    def __init__(
        self,
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        quantization="4bit",
        device_map="auto",
        torch_dtype=torch.float16
    ):
        """
        Initialize the EmailRephraser with a model and tokenizer.

        Args:
            model_name (str): Name of the model to load (e.g., "meta-llama/Llama-3.2-3B-Instruct").
            quantization (str): Quantization type ("4bit", "8bit", or None).
            device_map (str): Device mapping for the model (e.g., "auto").
            torch_dtype (torch.dtype): Data type for computations (e.g., torch.float16).
        """
        # Define quantization configuration
        self.quantization_config = None
        if quantization == "4bit":
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True
            )
        elif quantization == "8bit":
            self.quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=self.quantization_config,
            device_map=device_map
        )
        self.device = "cuda" if torch.cuda.is_available() and device_map == "auto" else "cpu"

        # Define prompt templates
        self.zero_shot_prompt_template = """<|im_start|>system
You are an assistant specialized in rephrasing emails. Your task is to take an original email and rephrase it while preserving its meaning, providing only the new subject and body in the following format:

**New Subject:** [New Subject]
**New Body:** [New Body]
<|im_end>

<|im_start|>user
Rephrase this email:

**Original Subject:** {subject}
**Original Body:** {body}
<|im_end>"""

        self.few_shot_prompt_template = """<|im_start|>system
You are an assistant specialized in rephrasing emails. Your task is to take an original email and rephrase it while preserving its meaning, providing only the new subject and body in the following format:

**New Subject:** [New Subject]
**New Body:** [New Body]

Here are examples to guide you:

**Example 1 Original Subject:** Urgent
**Example 1 Original Body:** Your account will be blocked from sending messages unless you complete the required email upgrade within the next 24 hours. Click the link below to unlock and upgrade your account to version 3.0.1. Ignoring this will result in suspension of your email services. Mail Service Team For: kevin@rocketinvestment.org
**Example 1 Output:**
**New Subject:** Account Upgrade Available
**New Body:** Dear Kevin, Upgrade to version 3.0.1 An upgrade is available for your account. Click the link below to unlock and upgrade your account. Thank you, Mail Service Team

**Example 2 Original Subject:** Security Alert: Unusual Login Attempt
**Example 2 Original Body:** We have detected an unauthorized login attempt on your account from a new device. If this was you, no further action is needed. If not, please verify your account immediately to prevent unauthorized access. Click the link below to secure your account: [Malicious Link] Failure to act within 12 hours will result in your account being locked. Security Team
**Example 2 Output:**
**New Subject:** Action Required: Verify Recent Login Activity
**New Body:** Dear User, We noticed a login attempt on your account from a new device. If this was you, no action is required. However, if you don’t recognize this activity, we recommend reviewing your security settings. For your convenience, you can check your recent login activity and update your security settings by following this link: [Legitimate-looking Link] Thank you for helping us keep your account secure. Support Team

**Example 3 Original Subject:** Payment Failure - Immediate Action Required
**Example 3 Original Body:** Dear Customer, Your recent payment attempt has failed due to an issue with your billing information. To avoid service disruption, please update your payment details immediately by clicking the link below: [Malicious Link] If we do not receive an update within 24 hours, your service will be suspended. Billing Department
**Example 3 Output:**
**New Subject:** Important: Update Your Billing Information
**New Body:** Dear Customer, We were unable to process your recent payment. This may be due to outdated billing details or an issue with your payment method. To ensure uninterrupted service, please review and update your payment information at your earliest convenience by visiting your account page: [Legitimate-looking Link] If you have already updated your details, please disregard this message. Billing Support Team
<|im_end>

<|im_start|>user
Rephrase this email:

**Original Subject:** {subject}
**Original Body:** {body}
<|im_end>"""

    def generate_rephrased_text(
        self,
        subject,
        body,
        prompt_type="zero_shot",
        max_new_tokens=200,
        temperature=0.6,
        top_p=0.9
    ):
        """
        Generate a rephrased email based on the input subject and body.

        Args:
            subject (str): Original email subject.
            body (str): Original email body.
            prompt_type (str): Type of prompt ("zero_shot" or "few_shot").
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Sampling temperature for creativity.
            top_p (float): Top-p sampling parameter for diversity.

        Returns:
            tuple: (new_subject, new_body) containing the rephrased subject and body.
        """
        # Select prompt template
        prompt_template = (
            self.few_shot_prompt_template if prompt_type == "few_shot" else self.zero_shot_prompt_template
        )

        # Format prompt with subject and body
        prompt = prompt_template.format(subject=subject, body=body)

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate output
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )

        # Decode generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract new subject and body
        try:
            new_subject = generated_text.rsplit("**New Subject:**", 1)[1].rsplit("**New Body:**", 1)[0].strip()
            new_body = generated_text.rsplit("**New Body:**", 1)[1].strip()
        except IndexError:
            raise ValueError("Failed to parse generated text. Ensure the model output follows the expected format.")

        return new_subject, new_body

    def __del__(self):
        """
        Clean up resources when the object is deleted.
        """
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
