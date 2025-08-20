#!/usr/bin/env python3
"""
Excel Underlying Data Processor - Separated NPZ Format
Each underlying code is saved as an independent NPZ file
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import argparse
from pathlib import Path

class SeparatedNPZProcessor:
    """Separated NPZ Data Processor"""
    
    def __init__(self, excel_file, output_dir='./underlying_npz'):
        """
        Initialize processor
        
        Args:
            excel_file (str): Excel file path
            output_dir (str): Output directory
        """
        self.excel_file = excel_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.underlying_data = {}
        self.statistics = {}
        
        print(f"Initializing separated NPZ data processor")
        print(f"Input file: {excel_file}")
        print(f"Output directory: {output_dir}")
    
    def read_excel_data(self):
        """Read all underlying data from Excel file"""
        print("\nStarting to read Excel file...")
        
        try:
            # Read all worksheet names
            excel_file = pd.ExcelFile(self.excel_file)
            all_sheets = excel_file.sheet_names
            
            # Filter out Setup and Weekly3 worksheets
            data_sheets = [sheet for sheet in all_sheets 
                          if sheet not in ['Setup', 'Weekly3']]
            
            print(f"Found {len(data_sheets)} underlying worksheets: {', '.join(data_sheets)}")
            
            # Process each worksheet
            for sheet_name in data_sheets:
                print(f"Processing {sheet_name}...")
                self._process_sheet(excel_file, sheet_name)
            
            print(f"\n✅ Successfully read data for {len(self.underlying_data)} underlyings")
            
        except Exception as e:
            print(f"❌ Failed to read Excel file: {e}")
            raise
    
    def _process_sheet(self, excel_file, sheet_name):
        """Process single worksheet"""
        try:
            # Read worksheet data, skip first 3 rows (headers and type definitions)
            df = pd.read_excel(excel_file, sheet_name=sheet_name, 
                             skiprows=3, header=None)
            
            # Set column names
            df.columns = ['datetime', 'price', 'volume']
            
            # Data cleaning
            df = df.dropna(subset=['datetime', 'price'])  # Remove null values
            
            # Convert data types
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
            
            # Clean again for failed conversions
            df = df.dropna(subset=['price'])
            
            if len(df) == 0:
                print(f"  ⚠️  {sheet_name}: No valid data")
                return
            
            # Convert to numpy array format
            timestamps = df['datetime'].astype('datetime64[s]').astype(np.int64)  # Unix timestamp
            prices = df['price'].astype(np.float64)
            volumes = df['volume'].astype(np.int64)
            
            # Store data
            self.underlying_data[sheet_name] = {
                'timestamps': timestamps,
                'prices': prices,
                'volumes': volumes
            }
            
            # Calculate statistics
            self._calculate_statistics(sheet_name, timestamps, prices, volumes)
            
            print(f"  ✅ {sheet_name}: {len(df)} records")
            
        except Exception as e:
            print(f"  ❌ {sheet_name} processing failed: {e}")
    
    def _calculate_statistics(self, code, timestamps, prices, volumes):
        """Calculate statistics"""
        
        # Calculate price statistics
        price_stats = {
            'min': float(prices.min()),
            'max': float(prices.max()),
            'mean': float(prices.mean()),
            'std': float(prices.std()),
            'median': float(np.median(prices))
        }
        
        # Calculate volume statistics
        volume_stats = {
            'min': int(volumes.min()),
            'max': int(volumes.max()),
            'mean': float(volumes.mean()),
            'total': int(volumes.sum()),
            'median': float(np.median(volumes))
        }
        
        # Time range
        time_stats = {
            'start_timestamp': int(timestamps.min()),
            'end_timestamp': int(timestamps.max()),
            'duration_days': int((timestamps.max() - timestamps.min()) / (24 * 3600))
        }
        
        # Calculate return statistics
        if len(prices) > 1:
            returns = np.diff(prices) / prices[:-1]
            return_stats = {
                'mean': float(returns.mean()),
                'std': float(returns.std()),
                'sharpe': float(returns.mean() / returns.std()) if returns.std() > 0 else 0,
                'min': float(returns.min()),
                'max': float(returns.max())
            }
        else:
            return_stats = {'mean': 0, 'std': 0, 'sharpe': 0, 'min': 0, 'max': 0}
        
        self.statistics[code] = {
            'record_count': len(prices),
            'price': price_stats,
            'volume': volume_stats,
            'time': time_stats,
            'returns': return_stats
        }
    
    def save_separated_npz_files(self):
        """Save each underlying as independent NPZ file"""
        print(f"\nStarting to save separated NPZ files to {self.output_dir}...")
        
        saved_files = []
        
        for code, data in self.underlying_data.items():
            try:
                # Build file path
                npz_file = self.output_dir / f'{code}.npz'
                
                # Prepare data to save
                save_data = {
                    # Raw data
                    'timestamps': data['timestamps'],
                    'prices': data['prices'],
                    'volumes': data['volumes'],
                    
                    # Metadata
                    'metadata_code': code,
                    'metadata_processed_at': datetime.now().isoformat(),
                    'metadata_source_file': str(self.excel_file),
                    'metadata_record_count': len(data['prices']),
                    
                    # Statistics
                    'stats_price_min': self.statistics[code]['price']['min'],
                    'stats_price_max': self.statistics[code]['price']['max'],
                    'stats_price_mean': self.statistics[code]['price']['mean'],
                    'stats_price_std': self.statistics[code]['price']['std'],
                    'stats_price_median': self.statistics[code]['price']['median'],
                    
                    'stats_volume_min': self.statistics[code]['volume']['min'],
                    'stats_volume_max': self.statistics[code]['volume']['max'],
                    'stats_volume_mean': self.statistics[code]['volume']['mean'],
                    'stats_volume_total': self.statistics[code]['volume']['total'],
                    'stats_volume_median': self.statistics[code]['volume']['median'],
                    
                    'stats_time_start': self.statistics[code]['time']['start_timestamp'],
                    'stats_time_end': self.statistics[code]['time']['end_timestamp'],
                    'stats_time_duration_days': self.statistics[code]['time']['duration_days'],
                    
                    'stats_returns_mean': self.statistics[code]['returns']['mean'],
                    'stats_returns_std': self.statistics[code]['returns']['std'],
                    'stats_returns_sharpe': self.statistics[code]['returns']['sharpe'],
                    'stats_returns_min': self.statistics[code]['returns']['min'],
                    'stats_returns_max': self.statistics[code]['returns']['max']
                }
                
                # Save NPZ file
                np.savez_compressed(npz_file, **save_data)
                saved_files.append(npz_file)
                
                # Display file info
                file_size = npz_file.stat().st_size / 1024  # KB
                print(f"  ✅ {code}: {npz_file.name} ({file_size:.1f} KB)")
                
            except Exception as e:
                print(f"  ❌ {code} save failed: {e}")
        
        # Save index file
        self._save_index_file(saved_files)
        
        print(f"\n✅ Total saved {len(saved_files)} NPZ files")
        return saved_files
    
    def _save_index_file(self, saved_files):
        """Save index file recording all underlying information"""
        index_file = self.output_dir / 'index.npz'
        
        # Collect all codes and basic information
        codes = list(self.underlying_data.keys())
        record_counts = [self.statistics[code]['record_count'] for code in codes]
        total_records = sum(record_counts)
        
        # Create index data
        index_data = {
            'codes': np.array(codes),
            'record_counts': np.array(record_counts),
            'total_codes': len(codes),
            'total_records': total_records,
            'processed_at': datetime.now().isoformat(),
            'source_file': str(self.excel_file),
            'file_names': np.array([f'{code}.npz' for code in codes])
        }
        
        # Add summary statistics
        for stat_type in ['price_min', 'price_max', 'price_mean', 'volume_total']:
            values = []
            for code in codes:
                if stat_type == 'price_min':
                    values.append(self.statistics[code]['price']['min'])
                elif stat_type == 'price_max':
                    values.append(self.statistics[code]['price']['max'])
                elif stat_type == 'price_mean':
                    values.append(self.statistics[code]['price']['mean'])
                elif stat_type == 'volume_total':
                    values.append(self.statistics[code]['volume']['total'])
            
            index_data[f'summary_{stat_type}'] = np.array(values)
        
        # Save index file
        np.savez_compressed(index_file, **index_data)
        print(f"  📋 Index file: {index_file.name}")
    
    def print_statistics(self):
        """Print detailed statistics"""
        print("\n" + "="*80)
        print("Data Statistics Summary")
        print("="*80)
        
        codes = list(self.statistics.keys())
        total_records = sum(self.statistics[code]['record_count'] for code in codes)
        
        print(f"Total: {len(codes)} underlyings, {total_records:,} records")
        print(f"Output directory: {self.output_dir}")
        print(f"File format: Independent NPZ file for each underlying\n")
        
        # Table header
        print(f"{'Code':<8} | {'Records':<8} | {'Start Date':<12} | {'End Date':<12} | {'Price Range':<20} | {'Total Volume':<12} | {'File Name':<12}")
        print("-" * 95)
        
        # Print statistics for each underlying
        for code in codes:
            stats = self.statistics[code]
            
            start_date = datetime.fromtimestamp(stats['time']['start_timestamp']).strftime('%Y-%m-%d')
            end_date = datetime.fromtimestamp(stats['time']['end_timestamp']).strftime('%Y-%m-%d')
            price_range = f"{stats['price']['min']:.3f}-{stats['price']['max']:.3f}"
            volume_total = f"{stats['volume']['total']:,}"
            file_name = f"{code}.npz"
            
            print(f"{code:<8} | {stats['record_count']:<8} | {start_date:<12} | {end_date:<12} | {price_range:<20} | {volume_total:<12} | {file_name:<12}")
        
        print("\n" + "="*80)
        
        # Detailed statistics
        print("\nDetailed Statistics:")
        for code in codes:
            stats = self.statistics[code]
            print(f"\n{code}:")
            print(f"  Records: {stats['record_count']:,}")
            print(f"  Time span: {stats['time']['duration_days']} days")
            print(f"  Price statistics: mean={stats['price']['mean']:.5f}, std={stats['price']['std']:.5f}")
            print(f"  Volume: total={stats['volume']['total']:,}, mean={stats['volume']['mean']:.2f}")
            print(f"  Returns: mean={stats['returns']['mean']:.6f}, volatility={stats['returns']['std']:.6f}, sharpe={stats['returns']['sharpe']:.4f}")
    
    def generate_usage_example(self):
        """Generate usage example code"""
        print("\n" + "="*60)
        print("Separated NPZ File Usage Example")
        print("="*60)
        
        codes = list(self.underlying_data.keys())
        example_code = f'''
# Separated NPZ File Usage Example
import numpy as np
from pathlib import Path
from datetime import datetime

# Data directory
DATA_DIR = "{self.output_dir}"

def load_index():
    """Load index file"""
    index_path = Path(DATA_DIR) / 'index.npz'
    if index_path.exists():
        return np.load(index_path, allow_pickle=True)
    else:
        return None

def get_available_codes():
    """Get all available underlying codes"""
    index = load_index()
    if index is not None:
        return [str(code) for code in index['codes']]
    else:
        # If no index file, scan directory
        data_dir = Path(DATA_DIR)
        return [f.stem for f in data_dir.glob('*.npz') if f.name != 'index.npz']

def load_underlying(code):
    """Load data for specific underlying"""
    npz_path = Path(DATA_DIR) / f'{{code}}.npz'
    
    if not npz_path.exists():
        raise FileNotFoundError(f"File does not exist: {{npz_path}}")
    
    data = np.load(npz_path, allow_pickle=True)
    
    timestamps = data['timestamps']
    prices = data['prices']
    volumes = data['volumes']
    
    return timestamps, prices, volumes

def get_statistics(code):
    """Get statistics"""
    npz_path = Path(DATA_DIR) / f'{{code}}.npz'
    
    if not npz_path.exists():
        raise FileNotFoundError(f"File does not exist: {{npz_path}}")
    
    data = np.load(npz_path, allow_pickle=True)
    
    stats = {{}}
    for key in data.files:
        if key.startswith('stats_'):
            stat_name = key[6:]  # Remove 'stats_' prefix
            stats[stat_name] = data[key].item()
    
    return stats

def get_metadata(code):
    """Get metadata"""
    npz_path = Path(DATA_DIR) / f'{{code}}.npz'
    
    if not npz_path.exists():
        raise FileNotFoundError(f"File does not exist: {{npz_path}}")
    
    data = np.load(npz_path, allow_pickle=True)
    
    metadata = {{}}
    for key in data.files:
        if key.startswith('metadata_'):
            meta_name = key[9:]  # Remove 'metadata_' prefix
            metadata[meta_name] = data[key].item()
    
    return metadata

# Usage example
if __name__ == "__main__":
    # Get all available codes
    codes = get_available_codes()
    print(f"Available underlying codes: {{codes}}")
    
    # Load index information
    index = load_index()
    if index is not None:
        print(f"\\nIndex information:")
        print(f"  Total underlyings: {{index['total_codes'].item()}}")
        print(f"  Total records: {{index['total_records'].item():,}}")
        print(f"  Processing time: {{index['processed_at'].item()}}")
    
    # Analyze first underlying
    if codes:
        code = codes[0]
        print(f"\\nAnalyzing {{code}}:")
        
        # Load data
        timestamps, prices, volumes = load_underlying(code)
        
        # Get statistics
        stats = get_statistics(code)
        metadata = get_metadata(code)
        
        print(f"  Data points: {{len(prices):,}}")
        print(f"  Price range: {{prices.min():.5f}} - {{prices.max():.5f}}")
        print(f"  Average price: {{prices.mean():.5f}}")
        print(f"  Total volume: {{volumes.sum():,}}")
        
        # Time range
        start_time = datetime.fromtimestamp(timestamps[0])
        end_time = datetime.fromtimestamp(timestamps[-1])
        print(f"  Time range: {{start_time}} to {{end_time}}")
        
        # Statistics
        print(f"  Price statistics: {{stats}}")
        
        # Return analysis
        if len(prices) > 1:
            returns = np.diff(prices) / prices[:-1]
            print(f"  Returns: mean={{returns.mean():.6f}}, std={{returns.std():.6f}}")

# Batch analysis example
def batch_analysis():
    """Batch analysis of all underlyings"""
    codes = get_available_codes()
    
    print(f"\\nBatch analysis results:")
    print(f"{'Code':<8} | {'Records':<8} | {'Mean Price':<10} | {'Volatility':<10} | {'Sharpe Ratio':<10}")
    print("-" * 60)
    
    for code in codes:
        try:
            timestamps, prices, volumes = load_underlying(code)
            stats = get_statistics(code)
            
            record_count = len(prices)
            mean_price = stats.get('price_mean', 0)
            returns_std = stats.get('returns_std', 0)
            sharpe = stats.get('returns_sharpe', 0)
            
            print(f"{{code:<8}} | {{record_count:<8,}} | {{mean_price:<10.5f}} | {{returns_std:<10.6f}} | {{sharpe:<10.4f}}")
            
        except Exception as e:
            print(f"{{code:<8}} | ERROR: {{e}}")

# Run batch analysis
batch_analysis()

print("\\nSeparated NPZ file usage example completed!")
'''
        
        print(example_code)
        
        # Save example code to file
        example_file = self.output_dir / 'usage_example.py'
        with open(example_file, 'w', encoding='utf-8') as f:
            f.write(example_code)
        
        print(f"\nExample code saved to: {example_file}")
    
    def process_all(self):
        """Execute complete processing workflow"""
        try:
            # 1. Read Excel data
            self.read_excel_data()
            
            # 2. Save separated NPZ files
            saved_files = self.save_separated_npz_files()
            
            if saved_files:
                # 3. Print statistics
                self.print_statistics()
                
                # 4. Generate usage example
                self.generate_usage_example()
                
                print(f"\n🎉 Processing completed!")
                print(f"📂 {len(saved_files)} NPZ files saved to: {self.output_dir.absolute()}")
                print(f"📋 Index file: index.npz")
                return True
            else:
                return False
            
        except Exception as e:
            print(f"\n❌ Processing failed: {e}")
            return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Process Excel underlying data and save as separated NPZ files')
    parser.add_argument('excel_file', help='Excel file path')
    parser.add_argument('-o', '--output', default='./underlying_npz', 
                       help='Output directory (default: ./underlying_npz)')
    
    args = parser.parse_args()
    
    # Check input file
    if not os.path.exists(args.excel_file):
        print(f"❌ File does not exist: {args.excel_file}")
        return 1
    
    # Create processor and execute
    processor = SeparatedNPZProcessor(args.excel_file, args.output)
    success = processor.process_all()
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())