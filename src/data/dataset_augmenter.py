import pandas as pd

class DatasetAugmenter:
    def __init__(self, dataset_path, rephraser):
        self.dataset = pd.read_csv(dataset_path)
        self.rephraser = rephraser

    def augment_dataset(self, output_path, mode="zero_shot"):
        """Augment the dataset using the specified mode (zero-shot or few-shot)."""
        augmented_data = []

        for _, row in self.dataset.iterrows():
            if mode == "zero_shot":
                prompt = self.rephraser.generate_zero_shot_prompt(row)
            elif mode == "few_shot":
                prompt = self.rephraser.generate_few_shot_prompt(row)
            else:
                raise ValueError("Invalid mode. Choose 'zero_shot' or 'few_shot'.")

            rephrased_email = self.rephraser.rephrase_email(prompt)
            augmented_data.append({
                "subject": rephrased_email.split("Subject: ")[1].split("\n")[0],
                "sender": row['sender'],
                "receiver": row['receiver'],
                "body": rephrased_email.split("Body: ")[1],
                "has_attachment": row['has_attachment'],
                "label": row['label'],
                "urls": row['urls'],
                "source": row['source'],
                "sender_clean": row['sender_clean'],
                "receiver_clean": row['receiver_clean'],
                "body_no_brackets": row['body_no_brackets'],
                "body_10_words": row['body_10_words']
            })

        augmented_df = pd.DataFrame(augmented_data)
        augmented_df.to_csv(output_path, index=False)
        print(f"Augmented dataset saved to {output_path}")