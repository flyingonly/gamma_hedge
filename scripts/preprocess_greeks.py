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
from common.greeks_config import GreeksPreprocessingConfig
import numpy as np

def process_weekly_code_with_stats(preprocessor: GreeksPreprocessor, weekly_code: str, force_reprocess: bool = True) -> dict:
    """
    Process a weekly code and return detailed statistics with input/output comparison
    
    Returns:
        dict: Statistics including input/output data counts and success rates
    """
    # Discover options for this weekly code
    options = preprocessor.discover_options_for_weekly(weekly_code)
    
    stats = {
        'options_count': 0,
        'total_input_data_points': 0,
        'total_output_data_points': 0,
        'avg_input_data_points': 0.0,
        'avg_output_data_points': 0.0,
        'overall_success_rate': 0.0,
        'option_details': {}
    }
    
    if not options:
        return stats
    
    # Process each option and collect input/output statistics
    for option_type, strike in options:
        try:
            # Get raw input data count first
            raw_file = preprocessor._get_raw_option_file_path(weekly_code, option_type, strike)
            input_count = 0
            if os.path.exists(raw_file):
                raw_data = np.load(raw_file, allow_pickle=True)
                input_count = len(raw_data['data']) if 'data' in raw_data else 0
            
            # Preprocess this specific option
            output_file = preprocessor.preprocess_single_option(weekly_code, option_type, strike, force_reprocess=force_reprocess)
            
            # Load the generated file and count output data points
            output_count = 0
            if os.path.exists(output_file):
                data = np.load(output_file)
                output_count = len(data['timestamps']) if 'timestamps' in data else 0
            
            # Calculate success rate for this option
            success_rate = (output_count / input_count * 100) if input_count > 0 else 0.0
            
            option_name = f"{option_type}_{strike}"
            stats['option_details'][option_name] = {
                'input_count': input_count,
                'output_count': output_count,
                'success_rate': success_rate
            }
            
            stats['total_input_data_points'] += input_count
            stats['total_output_data_points'] += output_count
            stats['options_count'] += 1
                
        except Exception as e:
            print(f"    Warning: Failed to process {option_type}_{strike}: {e}")
            continue
    
    # Calculate averages and overall success rate
    if stats['options_count'] > 0:
        stats['avg_input_data_points'] = stats['total_input_data_points'] / stats['options_count']
        stats['avg_output_data_points'] = stats['total_output_data_points'] / stats['options_count']
    
    if stats['total_input_data_points'] > 0:
        stats['overall_success_rate'] = stats['total_output_data_points'] / stats['total_input_data_points'] * 100
    
    return stats

def main(resume: bool = False):
    """Main preprocessing function"""
    print("Greeks Preprocessing Script")
    print("=" * 40)
    
    try:
        # Initialize preprocessor
        config = GreeksPreprocessingConfig()
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
        total_input_data_points = 0
        total_output_data_points = 0
        for i, weekly_code in enumerate(weekly_codes, 1):
            print(f"\n[{i}/{len(weekly_codes)}] Processing {weekly_code}...")
            
            try:
                # Get detailed statistics for this weekly code
                weekly_stats = process_weekly_code_with_stats(preprocessor, weekly_code, force_reprocess=not resume)
                total_processed += weekly_stats['options_count']
                total_input_data_points += weekly_stats['total_input_data_points']
                total_output_data_points += weekly_stats['total_output_data_points']
                
                print(f"  Successfully processed {weekly_stats['options_count']} options")
                print(f"  Input data points: {weekly_stats['total_input_data_points']}")
                print(f"  Output data points: {weekly_stats['total_output_data_points']}")
                print(f"  Overall success rate: {weekly_stats['overall_success_rate']:.1f}%")
                print(f"  Average input per option: {weekly_stats['avg_input_data_points']:.1f}")
                print(f"  Average output per option: {weekly_stats['avg_output_data_points']:.1f}")
                
            except Exception as e:
                print(f"  Error processing {weekly_code}: {e}")
                continue
        
        print(f"\nPreprocessing completed!")
        print(f"Total options processed: {total_processed}")
        print(f"Total input data points: {total_input_data_points}")
        print(f"Total output data points: {total_output_data_points}")
        overall_success_rate = (total_output_data_points/max(total_input_data_points,1)*100)
        print(f"Overall success rate: {overall_success_rate:.1f}%")
        print(f"Average input per option: {total_input_data_points/max(total_processed,1):.1f}")
        print(f"Average output per option: {total_output_data_points/max(total_processed,1):.1f}")
        print(f"Preprocessed files saved to: data/preprocessed_greeks/")
        
    except Exception as e:
        print(f"Error in main preprocessing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def preprocess_specific_codes(weekly_codes: List[str], resume: bool = False):
    """Preprocess specific weekly codes"""
    print(f"Greeks Preprocessing for codes: {weekly_codes}")
    print("=" * 40)
    
    try:
        config = GreeksPreprocessingConfig()
        preprocessor = GreeksPreprocessor(config)
        
        total_processed = 0
        total_input_data_points = 0
        total_output_data_points = 0
        for weekly_code in weekly_codes:
            print(f"\nProcessing {weekly_code}...")
            
            try:
                # Get detailed statistics for this weekly code
                weekly_stats = process_weekly_code_with_stats(preprocessor, weekly_code, force_reprocess=not resume)
                total_processed += weekly_stats['options_count']
                total_input_data_points += weekly_stats['total_input_data_points']
                total_output_data_points += weekly_stats['total_output_data_points']
                
                print(f"  Successfully processed {weekly_stats['options_count']} options")
                print(f"  Input data points: {weekly_stats['total_input_data_points']}")
                print(f"  Output data points: {weekly_stats['total_output_data_points']}")
                print(f"  Overall success rate: {weekly_stats['overall_success_rate']:.1f}%")
                print(f"  Average input per option: {weekly_stats['avg_input_data_points']:.1f}")
                print(f"  Average output per option: {weekly_stats['avg_output_data_points']:.1f}")
                
                # Show detailed breakdown for each option
                if weekly_stats['option_details']:
                    print(f"  Detailed breakdown (input → output):")
                    for option_name, details in weekly_stats['option_details'].items():
                        print(f"    {option_name}: {details['input_count']} → {details['output_count']} ({details['success_rate']:.1f}%)")
                
            except Exception as e:
                print(f"  Error processing {weekly_code}: {e}")
                continue
        
        print(f"\nProcessing completed!")
        print(f"Total options processed: {total_processed}")
        print(f"Total input data points: {total_input_data_points}")
        print(f"Total output data points: {total_output_data_points}")
        overall_success_rate = (total_output_data_points/max(total_input_data_points,1)*100)
        print(f"Overall success rate: {overall_success_rate:.1f}%")
        print(f"Average input per option: {total_input_data_points/max(total_processed,1):.1f}")
        print(f"Average output per option: {total_output_data_points/max(total_processed,1):.1f}")
        
    except Exception as e:
        print(f"Error in specific preprocessing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess Greeks for options")
    parser.add_argument('--codes', nargs='+', help='Specific weekly codes to process')
    parser.add_argument('--all', action='store_true', help='Process all available codes')
    parser.add_argument('--resume', action='store_true', help='Resume from existing cache (default: force reprocess all)')
    
    args = parser.parse_args()
    
    if args.codes:
        preprocess_specific_codes(args.codes, resume=args.resume)
    else:
        main(resume=args.resume)