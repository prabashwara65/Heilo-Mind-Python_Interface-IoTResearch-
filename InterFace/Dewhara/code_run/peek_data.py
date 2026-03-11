import pandas as pd
import csv

def peek_csv(file_path):
    print(f"--- Peek {file_path} ---")
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i >= 10: break
            print(row)

peek_csv("data/battery SOC.csv")
peek_csv("data/combined.csv")
