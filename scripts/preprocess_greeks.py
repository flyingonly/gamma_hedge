#!/usr/bin/env python3
"""
Greeks Preprocessing Script
===========================

Batch preprocesses Greeks (Delta, Gamma, Theta, Vega) for all available options.
Run this script to generate preprocessed Greeks files for faster dataset loading.
"""

import sys
import os
from typing import List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.greeks_preprocessor import GreeksPreprocessor
from delta_hedge.config import DeltaHedgeConfig

def main():
    """Main preprocessing function"""
    print("Greeks Preprocessing Script")
    print("=" * 40)
    
    try:
        # Initialize preprocessor
        config = DeltaHedgeConfig()
        preprocessor = GreeksPreprocessor(config)
        
        # Discover all available weekly codes
        print("Discovering available weekly codes...")
        weekly_codes = preprocessor.discover_weekly_codes()
        print(f"Found {len(weekly_codes)} weekly codes: {weekly_codes}")
        
        if not weekly_codes:
            print("No weekly codes found. Please check csv_process/weekly_options_data/ directory.")
            return
        
        # Process all weekly codes
        total_processed = 0
        for i, weekly_code in enumerate(weekly_codes, 1):
            print(f"\n[{i}/{len(weekly_codes)}] Processing {weekly_code}...")
            
            try:
                results = preprocessor.batch_preprocess([weekly_code])
                processed_count = len(results.get(weekly_code, []))
                total_processed += processed_count
                print(f"  Successfully processed {processed_count} options")
                
            except Exception as e:
                print(f"  Error processing {weekly_code}: {e}")
                continue
        
        print(f"\nPreprocessing completed!")
        print(f"Total options processed: {total_processed}")
        print(f"Preprocessed files saved to: data/preprocessed_greeks/")
        
    except Exception as e:
        print(f"Error in main preprocessing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def preprocess_specific_codes(weekly_codes: List[str]):
    """Preprocess specific weekly codes"""
    print(f"Greeks Preprocessing for codes: {weekly_codes}")
    print("=" * 40)
    
    try:
        config = DeltaHedgeConfig()
        preprocessor = GreeksPreprocessor(config)
        
        total_processed = 0
        for weekly_code in weekly_codes:
            print(f"\nProcessing {weekly_code}...")
            
            try:
                results = preprocessor.batch_preprocess([weekly_code])
                processed_count = len(results.get(weekly_code, []))
                total_processed += processed_count
                print(f"  Successfully processed {processed_count} options")
                
            except Exception as e:
                print(f"  Error processing {weekly_code}: {e}")
                continue
        
        print(f"\nProcessing completed!")
        print(f"Total options processed: {total_processed}")
        
    except Exception as e:
        print(f"Error in specific preprocessing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess Greeks for options")
    parser.add_argument('--codes', nargs='+', help='Specific weekly codes to process')
    parser.add_argument('--all', action='store_true', help='Process all available codes')
    
    args = parser.parse_args()
    
    if args.codes:
        preprocess_specific_codes(args.codes)
    else:
        main()