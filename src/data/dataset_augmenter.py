import pandas as pd

class DatasetAugmenter:
    def __init__(self, dataset_path, rephraser):
        self.dataset = pd.read_csv(dataset_path)
        self.rephraser = rephraser
        self.augmented_data = []

    def augment_dataset(self, output_path, mode="zero_shot"):
        """Augment the dataset using the specified mode (zero-shot or few-shot)."""
        self.augmented_data = []

        for _, row in self.dataset.iterrows():
            if mode == "zero_shot":
                prompt = self.rephraser.generate_zero_shot_prompt(row)
            elif mode == "few_shot":
                prompt = self.rephraser.generate_few_shot_prompt(row)
            else:
                raise ValueError("Invalid mode. Choose 'zero_shot' or 'few_shot'.")

            rephrased_email = self.rephraser.rephrase_email(prompt)
            self.augmented_data.append({
                "subject": rephrased_email.split("Subject: ")[1].split("\n")[0],
                "sender": row['sender'],
                "receiver": row['receiver'],
                "body": rephrased_email.split("Body: ")[1]
            })

        augmented_df = pd.DataFrame(self.augmented_data)
        augmented_df.to_csv(output_path, index=False)
        print(f"Augmented dataset saved to {output_path}")
    
    def print_augmented_emails(self, n=1):
        """Print the first n augmented emails."""
        for i in range(n):
            print(f"Original email {i+1}:")
            print("Subject:", self.dataset.iloc[i]['subject'])
            print("Body:", self.dataset.iloc[i]['body'])
            print("\nAugmented email:")
            print("Subject:", self.augmented_data[i]['subject'])
            print("Body:", self.augmented_data[i]['body'])
            print("\n" + "-"*50 + "\n")