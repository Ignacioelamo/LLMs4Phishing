import pandas as pd
import os
import argparse

def process_csv(filepath):
    """
    Reads a CSV file, selects specific columns, removes rows with label 0,
    and drops the 'label' column from the final result.

    Args:
        filepath: Path to the CSV file.

    Returns:
        A cleaned pandas DataFrame or None if an error occurs.
    """
    try:
        df = pd.read_csv(filepath)
        df = df[['subject', 'body', 'label']]
        print(f"\nProcessing: {os.path.basename(filepath)}")
        print("Label distribution:\n", df['label'].value_counts())

        df = df[df['label'] != 0]
        df = df.drop(columns=['label'])  # Drop the 'label' column
        return df

    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return None
    except KeyError as e:
        print(f"❌ Missing column: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error in {filepath}: {e}")
        return None

def save_processed(df, original_path, output_dir):
    """
    Saves the cleaned DataFrame to the specified output directory
    with a new filename indicating it has been cleaned.
    """
    filename = os.path.basename(original_path)
    filename_clean = os.path.splitext(filename)[0] + '_clean.csv'
    output_path = os.path.join(output_dir, filename_clean)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="""
CSV cleaning script:
- Removes rows with label = 0
- Drops the 'label' column
- Keeps only the 'subject' and 'body' columns
Can be used on a single file or an entire directory of CSV files.
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("path", help="Path to a CSV file or a directory containing CSV files.")
    parser.add_argument(
        "--output",
        default="../../data/cleaned",
        help="Directory to save cleaned files (default: ../../data/cleaned)"
    )

    args = parser.parse_args()
    input_path = args.path
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(input_path):
        df = process_csv(input_path)
        if df is not None:
            save_processed(df, input_path, output_dir)

    elif os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            if filename.endswith(".csv"):
                full_path = os.path.join(input_path, filename)
                df = process_csv(full_path)
                if df is not None:
                    save_processed(df, full_path, output_dir)
    else:
        print("❌ Error: The specified path is neither a valid file nor a directory.")

if __name__ == "__main__":
    main()

