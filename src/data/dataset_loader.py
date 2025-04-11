import pandas as pd
from typing import Optional

class DatasetLoader:
    """Handles loading and preprocessing of datasets."""
    
    def __init__(self, dataset_path: str, subject_column: str, body_column: str):
        self.dataset_path = dataset_path
        self.subject_column = subject_column
        self.body_column = body_column
        self.data = None
    
    def load_data(self) -> pd.DataFrame:
        """Load the dataset from the specified path."""
        try:
            self.data = pd.read_csv(self.dataset_path)
            return self.data
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
    
    def validate_columns(self) -> bool:
        """Validate that required columns exist in the dataset."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        required_columns = [self.subject_column, self.body_column]
        missing = [col for col in required_columns if col not in self.data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return True
    
    def get_emails(self) -> list:
        """Return a list of (subject, body) tuples."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        self.validate_columns()
        return [(row[self.subject_column], row[self.body_column]) for _, row in self.data.iterrows()]