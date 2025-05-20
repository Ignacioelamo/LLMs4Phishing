import os
import sys
import argparse
import pandas as pd
from pathlib import Path

def create_eml(subject, body, output_path, index):
    content = f"""From: test@example.com
To: recipient@example.com
Subject: {subject}
Date: Thu, 09 May 2025 12:00:00 +0000
MIME-Version: 1.0
Content-Type: text/plain; charset="UTF-8"
Content-Transfer-Encoding: 7bit

{body}
"""
    filename = f"email_{index}.eml"
    filepath = output_path / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

def process_csv(file_path, tipo, output_base):
    dataset_name = file_path.stem
    output_folder_name = f"{dataset_name}_{tipo}"
    output_path = output_base / output_folder_name

    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(file_path)
    if 'subject' not in df.columns or 'body' not in df.columns:
        print(f"[!] Saltando {file_path} (faltan columnas 'subject' o 'body')")
        return

    for i, row in df.iterrows():
        create_eml(row['subject'], row['body'], output_path, f"{dataset_name}_{i}")

def main():
    parser = argparse.ArgumentParser(description="Generador de .eml desde CSV")
    parser.add_argument("ruta", help="Ruta a un archivo CSV o directorio con CSVs")
    parser.add_argument("--tipo", required=True, choices=["original", "refraseado"],
                        help="Tipo de dataset: original o refraseado")
    parser.add_argument("--output", default=".", help="Directorio base de salida (opcional)")

    args = parser.parse_args()
    input_path = Path(args.ruta)
    output_base = Path(args.output)

    if input_path.is_file():
        process_csv(input_path, args.tipo, output_base)
    elif input_path.is_dir():
        for file in input_path.glob("*.csv"):
            process_csv(file, args.tipo, output_base)
    else:
        print(f"[!] Ruta no válida: {input_path}")

if __name__ == "__main__":
    main()
