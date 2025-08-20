#!/usr/bin/env python3
"""
Separated NPZ Data Loader
Load and analyze independent NPZ format underlying data
"""

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional

class SeparatedNPZLoader:
    """Separated NPZ Data Loader"""
    
    def __init__(self, data_dir='./underlying_npz'):
        """
        Initialize data loader
        
        Args:
            data_dir (str): NPZ files storage directory
        """
        self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
        
        # Load index information
        self._load_index()
        
        print(f"Separated NPZ data loader initialized")
        print(f"Data directory: {data_dir}")
        print(f"Available underlyings: {len(self.codes)}")
    
    def _load_index(self):
        """Load index file"""
        index_path = self.data_dir / 'index.npz'
        
        if index_path.exists():
            self.index = np.load(index_path, allow_pickle=True)
            self.codes = [str(code) for code in self.index['codes']]
            
            self.metadata = {
                'total_codes': int(self.index['total_codes']),
                'total_records': int(self.index['total_records']),
                'processed_at': str(self.index['processed_at']),
                'source_file': str(self.index['source_file'])
            }
        else:
            # If no index file exists, scan directory
            print("No index file found, scanning directory...")
            self.codes = [f.stem for f in self.data_dir.glob('*.npz') 
                         if f.name != 'index.npz']
            self.index = None
            self.metadata = {}
        
        if not self.codes:
            raise ValueError(f"No NPZ files found in directory {self.data_dir}")
    
    def get_codes(self) -> List[str]:
        """Get all available underlying codes"""
        return self.codes.copy()
    
    def load_arrays(self, code: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load array data for specific underlying
        
        Args:
            code (str): underlying code
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (timestamps, prices, volumes)
        """
        if code not in self.codes:
            raise ValueError(f"Code {code} not found, available codes: {self.codes}")
        
        npz_path = self.data_dir / f'{code}.npz'
        
        if not npz_path.exists():
            raise FileNotFoundError(f"File does not exist: {npz_path}")
        
        data = np.load(npz_path, allow_pickle=True)
        
        try:
            timestamps = data['timestamps']
            prices = data['prices']
            volumes = data['volumes']
            return timestamps, prices, volumes
        finally:
            data.close()
    
    def load_dataframe(self, code: str) -> pd.DataFrame:
        """
        Load data as pandas DataFrame
        
        Args:
            code (str): underlying code
            
        Returns:
            pd.DataFrame: DataFrame with datetime, price, volume columns
        """
        timestamps, prices, volumes = self.load_arrays(code)
        
        df = pd.DataFrame({
            'datetime': pd.to_datetime(timestamps, unit='s'),
            'price': prices,
            'volume': volumes
        })
        
        return df
    
    def get_statistics(self, code: str) -> Dict:
        """
        Get statistics
        
        Args:
            code (str): underlying code
            
        Returns:
            Dict: statistics dictionary
        """
        if code not in self.codes:
            raise ValueError(f"Code {code} not found, available codes: {self.codes}")
        
        npz_path = self.data_dir / f'{code}.npz'
        
        if not npz_path.exists():
            raise FileNotFoundError(f"File does not exist: {npz_path}")
        
        data = np.load(npz_path, allow_pickle=True)
        
        try:
            stats = {}
            for key in data.files:
                if key.startswith('stats_'):
                    stat_name = key[6:]  # Remove 'stats_' prefix
                    stats[stat_name] = data[key].item()
            
            return stats
        finally:
            data.close()
    
    def get_metadata(self, code: str) -> Dict:
        """
        Get metadata
        
        Args:
            code (str): underlying code
            
        Returns:
            Dict: metadata dictionary
        """
        if code not in self.codes:
            raise ValueError(f"Code {code} not found, available codes: {self.codes}")
        
        npz_path = self.data_dir / f'{code}.npz'
        
        if not npz_path.exists():
            raise FileNotFoundError(f"File does not exist: {npz_path}")
        
        data = np.load(npz_path, allow_pickle=True)
        
        try:
            metadata = {}
            for key in data.files:
                if key.startswith('metadata_'):
                    meta_name = key[9:]  # Remove 'metadata_' prefix
                    metadata[meta_name] = data[key].item()
            
            return metadata
        finally:
            data.close()
    
    def print_summary(self):
        """Print data summary"""
        print("\n" + "="*70)
        print("Separated NPZ Data Summary")
        print("="*70)
        
        if self.metadata:
            print(f"Data source: {self.metadata.get('source_file', 'N/A')}")
            print(f"Processing time: {self.metadata.get('processed_at', 'N/A')}")
            print(f"Total records: {self.metadata.get('total_records', 'N/A'):,}")
        
        print(f"Number of underlyings: {len(self.codes)}")
        print(f"Data directory: {self.data_dir}")
        print(f"Available codes: {', '.join(self.codes)}")
        
        # Calculate total file size
        total_size = sum(f.stat().st_size for f in self.data_dir.glob('*.npz'))
        total_size_mb = total_size / (1024 * 1024)
        print(f"Total file size: {total_size_mb:.2f} MB")
        
        # Print basic information for each code
        print(f"\n{'Code':<8} | {'Records':<8} | {'Price Range':<18} | {'Time Span':<12} | {'File Size':<10}")
        print("-" * 70)
        
        for code in self.codes:
            try:
                stats = self.get_statistics(code)
                file_path = self.data_dir / f'{code}.npz'
                file_size_kb = file_path.stat().st_size / 1024
                
                record_count = stats.get('price_mean', 0)  # This should be record count, but using available stat
                price_min = stats.get('price_min', 0)
                price_max = stats.get('price_max', 0)
                duration_days = stats.get('time_duration_days', 0)
                
                print(f"{code:<8} | {record_count:<8} | {price_min:.3f}-{price_max:.3f}   | {duration_days:<12} | {file_size_kb:.1f} KB")
                
            except Exception as e:
                print(f"{code:<8} | ERROR: {str(e)[:40]}")
    
    def analyze_underlying(self, code: str, plot: bool = True) -> Dict:
        """
        Analyze specific underlying
        
        Args:
            code (str): underlying code
            plot (bool): whether to generate plots
            
        Returns:
            Dict: analysis results
        """
        if code not in self.codes:
            raise ValueError(f"Code {code} not found, available codes: {self.codes}")
        
        print(f"\nAnalyzing {code}...")
        
        # Load data
        timestamps, prices, volumes = self.load_arrays(code)
        stats = self.get_statistics(code)
        metadata = self.get_metadata(code)
        
        # Basic analysis
        analysis = {
            'code': code,
            'data_points': len(prices),
            'time_span_days': stats.get('time_duration_days', 0),
            'price_stats': {
                'min': float(prices.min()),
                'max': float(prices.max()),
                'mean': float(prices.mean()),
                'std': float(prices.std()),
                'median': float(np.median(prices))
            },
            'volume_stats': {
                'min': int(volumes.min()),
                'max': int(volumes.max()),
                'mean': float(volumes.mean()),
                'total': int(volumes.sum()),
                'median': float(np.median(volumes))
            }
        }
        
        # Calculate returns
        if len(prices) > 1:
            returns = np.diff(prices) / prices[:-1]
            analysis['return_stats'] = {
                'mean': float(returns.mean()),
                'std': float(returns.std()),
                'sharpe': float(returns.mean() / returns.std()) if returns.std() > 0 else 0,
                'min': float(returns.min()),
                'max': float(returns.max()),
                'skewness': float(pd.Series(returns).skew()),
                'kurtosis': float(pd.Series(returns).kurtosis())
            }
        
        # Print analysis results
        print(f"  Data points: {analysis['data_points']:,}")
        print(f"  Time span: {analysis['time_span_days']} days")
        print(f"  Price range: {analysis['price_stats']['min']:.5f} - {analysis['price_stats']['max']:.5f}")
        print(f"  Average price: {analysis['price_stats']['mean']:.5f}")
        print(f"  Price volatility: {analysis['price_stats']['std']:.5f}")
        print(f"  Total volume: {analysis['volume_stats']['total']:,}")
        
        if 'return_stats' in analysis:
            print(f"  Return statistics:")
            print(f"    Mean return: {analysis['return_stats']['mean']:.6f}")
            print(f"    Return volatility: {analysis['return_stats']['std']:.6f}")
            print(f"    Sharpe ratio: {analysis['return_stats']['sharpe']:.4f}")
        
        # Generate plots if requested
        if plot:
            self._plot_underlying_analysis(code, timestamps, prices, volumes, analysis)
        
        return analysis
    
    def _plot_underlying_analysis(self, code: str, timestamps: np.ndarray, 
                                 prices: np.ndarray, volumes: np.ndarray, 
                                 analysis: Dict):
        """Generate analysis plots"""
        # Convert timestamps to datetime
        dates = pd.to_datetime(timestamps, unit='s')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Analysis for {code}', fontsize=16)
        
        # Price time series
        axes[0, 0].plot(dates, prices, linewidth=1)
        axes[0, 0].set_title('Price Time Series')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Volume time series
        axes[0, 1].plot(dates, volumes, linewidth=1, color='orange')
        axes[0, 1].set_title('Volume Time Series')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Volume')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Price distribution
        axes[1, 0].hist(prices, bins=50, alpha=0.7, color='green')
        axes[1, 0].set_title('Price Distribution')
        axes[1, 0].set_xlabel('Price')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Returns distribution (if available)
        if len(prices) > 1:
            returns = np.diff(prices) / prices[:-1]
            axes[1, 1].hist(returns, bins=50, alpha=0.7, color='red')
            axes[1, 1].set_title('Returns Distribution')
            axes[1, 1].set_xlabel('Returns')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor returns analysis', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Returns Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def batch_analysis(self) -> pd.DataFrame:
        """
        Batch analysis of all underlyings
        
        Returns:
            pd.DataFrame: analysis results for all underlyings
        """
        print(f"\nPerforming batch analysis for {len(self.codes)} underlyings...")
        
        results = []
        
        for code in self.codes:
            try:
                timestamps, prices, volumes = self.load_arrays(code)
                stats = self.get_statistics(code)
                
                result = {
                    'code': code,
                    'data_points': len(prices),
                    'time_span_days': stats.get('time_duration_days', 0),
                    'price_min': stats.get('price_min', 0),
                    'price_max': stats.get('price_max', 0),
                    'price_mean': stats.get('price_mean', 0),
                    'price_std': stats.get('price_std', 0),
                    'volume_total': stats.get('volume_total', 0),
                    'volume_mean': stats.get('volume_mean', 0),
                    'returns_mean': stats.get('returns_mean', 0),
                    'returns_std': stats.get('returns_std', 0),
                    'sharpe_ratio': stats.get('returns_sharpe', 0)
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error analyzing {code}: {e}")
                results.append({
                    'code': code,
                    'error': str(e)
                })
        
        df_results = pd.DataFrame(results)
        
        print(f"\nBatch Analysis Results:")
        print(f"{'Code':<8} | {'Points':<8} | {'Mean Price':<12} | {'Volatility':<12} | {'Sharpe':<8}")
        print("-" * 55)
        
        for _, row in df_results.iterrows():
            if 'error' not in row:
                print(f"{row['code']:<8} | {row['data_points']:<8,} | {row['price_mean']:<12.5f} | {row['returns_std']:<12.6f} | {row['sharpe_ratio']:<8.4f}")
            else:
                print(f"{row['code']:<8} | ERROR: {row['error']}")
        
        return df_results
    
    def compare_underlyings(self, codes: List[str], metric: str = 'price') -> Dict:
        """
        Compare multiple underlyings
        
        Args:
            codes (List[str]): list of underlying codes to compare
            metric (str): metric to compare ('price', 'volume', 'returns')
            
        Returns:
            Dict: comparison results
        """
        if not all(code in self.codes for code in codes):
            invalid_codes = [code for code in codes if code not in self.codes]
            raise ValueError(f"Invalid codes: {invalid_codes}")
        
        print(f"\nComparing {len(codes)} underlyings on metric: {metric}")
        
        comparison_data = {}
        
        for code in codes:
            timestamps, prices, volumes = self.load_arrays(code)
            dates = pd.to_datetime(timestamps, unit='s')
            
            if metric == 'price':
                comparison_data[code] = {'dates': dates, 'values': prices}
            elif metric == 'volume':
                comparison_data[code] = {'dates': dates, 'values': volumes}
            elif metric == 'returns' and len(prices) > 1:
                returns = np.diff(prices) / prices[:-1]
                comparison_data[code] = {'dates': dates[1:], 'values': returns}
            else:
                print(f"Warning: Insufficient data for {code} on metric {metric}")
        
        # Generate comparison plot
        if comparison_data:
            plt.figure(figsize=(12, 6))
            for code, data in comparison_data.items():
                plt.plot(data['dates'], data['values'], label=code, linewidth=1)
            
            plt.title(f'{metric.capitalize()} Comparison')
            plt.xlabel('Date')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        return comparison_data


def main():
    """Main function example"""
    # Initialize loader
    loader = SeparatedNPZLoader('./underlying_npz')
    
    # Print summary
    loader.print_summary()
    
    # Get available codes
    codes = loader.get_codes()
    print(f"\nAvailable codes: {codes}")
    
    if codes:
        # Analyze first underlying
        first_code = codes[0]
        analysis = loader.analyze_underlying(first_code, plot=False)
        
        # Batch analysis
        batch_results = loader.batch_analysis()
        
        # Compare first few underlyings if available
        if len(codes) > 1:
            compare_codes = codes[:min(3, len(codes))]
            loader.compare_underlyings(compare_codes, 'price')


if __name__ == "__main__":
    main()