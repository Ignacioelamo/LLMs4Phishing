from EmailRephraser import EmailRephraser

class FewShotRephraser(EmailRephraser):
    def __init__(self, model_name, few_shot_examples):
        super().__init__(model_name)
        self.few_shot_examples = few_shot_examples

    def generate_few_shot_prompt(self, email_data):
        """Generate a few-shot prompt for rephrasing."""
        prompt = f"""
        Below are some examples of rephrased phishing emails. Please rephrase the following email in a similar way.

        Examples:
        {self.few_shot_examples}

        Original Email:
        Subject: {email_data['subject']}
        Sender: {email_data['sender']}
        Receiver: {email_data['receiver']}
        Body: {email_data['body']}

        Rephrased Email:
        """
        return prompt