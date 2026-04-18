#!/usr/bin/env python3
"""
Fix LSWW29 time annotation by adding minutes_since_start column
"""
import pandas as pd
import os

def fix_time_in_file(input_file, output_file):
    """Add minutes_since_start column based on row index (5 min intervals)"""
    print(f"Processing: {input_file}")

    # Read CSV
    df = pd.read_csv(input_file, encoding='utf-8-sig')

    # Add minutes_since_start (each row is 5 minutes apart)
    df['minutes_since_start'] = [i * 5 for i in range(len(df))]

    # Save
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"  Rows: {len(df)}, Time range: 0 to {(len(df)-1)*5} minutes")
    print(f"  Output: {output_file}")
    print(f"  Columns: {list(df.columns)}")

    return df

def main():
    """Fix all LSWW29 split files"""

    print("="*60)
    print("Fixing LSWW29 Time Annotation")
    print("="*60)

    for split in ['train', 'val', 'test']:
        input_file = f'data/lsww_29c_split/{split}.csv'
        output_file = f'data/lsww_29c_split/{split}_fixed.csv'

        if os.path.exists(input_file):
            fix_time_in_file(input_file, output_file)
        else:
            print(f"Warning: {input_file} not found")

        print()

    print("="*60)
    print("All files fixed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
