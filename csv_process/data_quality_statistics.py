#!/usr/bin/env python3
"""
Data Quality Statistics Tool
============================

Comprehensive data quality analysis for underlying and options data.
Generates detailed statistics and tables for data coverage, quality, and completeness.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataQualityStatistics:
    """
    Data Quality Statistics analyzer for gamma hedge project.
    Analyzes underlying data, options data, and preprocessed Greeks.
    """
    
    def __init__(self, 
                 underlying_path: str = "csv_process/underlying_npz",
                 options_path: str = "csv_process/weekly_options_data", 
                 greeks_path: str = "data/preprocessed_greeks"):
        """
        Initialize data quality analyzer.
        
        Args:
            underlying_path: Path to underlying NPZ data
            options_path: Path to weekly options data
            greeks_path: Path to preprocessed Greeks data
        """
        self.underlying_path = Path(underlying_path)
        self.options_path = Path(options_path)
        self.greeks_path = Path(greeks_path)
        
        # Results storage
        self.underlying_stats = {}
        self.options_stats = {}
        self.greeks_stats = {}
        self.summary_stats = {}
        
        logger.info("Data Quality Statistics initialized")
        logger.info(f"Underlying path: {self.underlying_path}")
        logger.info(f"Options path: {self.options_path}")
        logger.info(f"Greeks path: {self.greeks_path}")
    
    def scan_underlying_data(self) -> Dict[str, Any]:
        """
        Scan and analyze underlying data files.
        
        Returns:
            Dict with underlying data statistics
        """
        logger.info("Scanning underlying data...")
        
        underlying_files = list(self.underlying_path.glob("*.npz"))
        underlying_files = [f for f in underlying_files if f.name != "index.npz"]
        
        results = {
            'total_files': len(underlying_files),
            'files': {},
            'summary': {
                'total_data_points': 0,
                'total_file_size_mb': 0,
                'avg_data_points': 0,
                'date_range': {'earliest': None, 'latest': None}
            }
        }
        
        total_points = 0
        total_size = 0
        all_dates = []
        
        for file_path in underlying_files:
            code = file_path.stem
            try:
                # Load NPZ file
                data = np.load(file_path, allow_pickle=True)
                
                # Extract basic data
                timestamps = data.get('timestamps', np.array([]))
                prices = data.get('prices', np.array([]))
                volumes = data.get('volumes', np.array([]))
                
                # File statistics
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                data_points = len(prices)
                
                # Time analysis
                if len(timestamps) > 0:
                    start_time = datetime.fromtimestamp(timestamps[0])
                    end_time = datetime.fromtimestamp(timestamps[-1])
                    duration_days = (end_time - start_time).days
                    all_dates.extend([start_time, end_time])
                else:
                    start_time = end_time = None
                    duration_days = 0
                
                # Price statistics
                price_stats = {}
                if len(prices) > 0:
                    price_stats = {
                        'min': float(prices.min()),
                        'max': float(prices.max()),
                        'mean': float(prices.mean()),
                        'std': float(prices.std()),
                        'median': float(np.median(prices))
                    }
                
                # Volume statistics
                volume_stats = {}
                if len(volumes) > 0:
                    volume_stats = {
                        'min': int(volumes.min()),
                        'max': int(volumes.max()),
                        'mean': float(volumes.mean()),
                        'total': int(volumes.sum()),
                        'median': float(np.median(volumes))
                    }
                
                # Returns analysis
                returns_stats = {}
                if len(prices) > 1:
                    returns = np.diff(prices) / prices[:-1]
                    returns_stats = {
                        'mean': float(returns.mean()),
                        'std': float(returns.std()),
                        'sharpe': float(returns.mean() / returns.std()) if returns.std() > 0 else 0,
                        'min': float(returns.min()),
                        'max': float(returns.max())
                    }
                
                # Store results
                results['files'][code] = {
                    'data_points': data_points,
                    'file_size_mb': file_size_mb,
                    'duration_days': duration_days,
                    'start_time': start_time.isoformat() if start_time else None,
                    'end_time': end_time.isoformat() if end_time else None,
                    'price_stats': price_stats,
                    'volume_stats': volume_stats,
                    'returns_stats': returns_stats,
                    'fields_available': list(data.files)
                }
                
                # Accumulate totals
                total_points += data_points
                total_size += file_size_mb
                
                data.close()
                
            except Exception as e:
                logger.error(f"Error processing {code}: {e}")
                results['files'][code] = {'error': str(e)}
        
        # Calculate summary statistics
        if results['total_files'] > 0:
            results['summary']['total_data_points'] = total_points
            results['summary']['total_file_size_mb'] = total_size
            results['summary']['avg_data_points'] = total_points / results['total_files']
            
            if all_dates:
                results['summary']['date_range']['earliest'] = min(all_dates).isoformat()
                results['summary']['date_range']['latest'] = max(all_dates).isoformat()
        
        self.underlying_stats = results
        logger.info(f"Found {results['total_files']} underlying files with {total_points:,} total data points")
        return results
    
    def scan_options_data(self) -> Dict[str, Any]:
        """
        Scan and analyze options data files.
        
        Returns:
            Dict with options data statistics
        """
        logger.info("Scanning options data...")
        
        weekly_codes = [d for d in self.options_path.iterdir() if d.is_dir()]
        
        results = {
            'total_weekly_codes': len(weekly_codes),
            'weekly_codes': {},
            'summary': {
                'total_options': 0,
                'total_calls': 0,
                'total_puts': 0,
                'total_file_size_mb': 0,
                'strike_range': {'min': float('inf'), 'max': float('-inf')},
                'avg_options_per_code': 0
            }
        }
        
        total_options = 0
        total_calls = 0
        total_puts = 0
        total_size = 0
        all_strikes = []
        
        for code_dir in weekly_codes:
            code = code_dir.name
            option_files = list(code_dir.glob("*.npz"))
            
            code_stats = {
                'total_options': len(option_files),
                'calls': 0,
                'puts': 0,
                'strikes': [],
                'file_size_mb': 0,
                'options': {}
            }
            
            for option_file in option_files:
                try:
                    # Parse option details from filename
                    file_name = option_file.stem
                    if file_name.startswith('CALL_'):
                        option_type = 'CALL'
                        strike_str = file_name[5:]
                        code_stats['calls'] += 1
                    elif file_name.startswith('PUT_'):
                        option_type = 'PUT'
                        strike_str = file_name[4:]
                        code_stats['puts'] += 1
                    else:
                        continue
                    
                    try:
                        strike = float(strike_str)
                        code_stats['strikes'].append(strike)
                        all_strikes.append(strike)
                    except ValueError:
                        logger.warning(f"Could not parse strike from {file_name}")
                        continue
                    
                    # File size
                    file_size_mb = option_file.stat().st_size / (1024 * 1024)
                    code_stats['file_size_mb'] += file_size_mb
                    
                    # Load NPZ data for detailed analysis
                    data = np.load(option_file, allow_pickle=True)
                    
                    # Extract option data details
                    option_stats = {
                        'type': option_type,
                        'strike': strike,
                        'file_size_mb': file_size_mb,
                        'fields_available': list(data.files)
                    }
                    
                    # Get data points from data_count field or data array
                    if 'data_count' in data.files:
                        option_stats['data_points'] = int(data['data_count'].item())
                    elif 'data' in data.files:
                        data_array = data['data']
                        if hasattr(data_array, 'shape') and len(data_array.shape) > 0:
                            option_stats['data_points'] = data_array.shape[0]
                        else:
                            option_stats['data_points'] = 0
                    else:
                        option_stats['data_points'] = 0
                    
                    # Try to get time information from data array
                    if 'data' in data.files and option_stats['data_points'] > 0:
                        data_array = data['data']
                        if hasattr(data_array, 'shape') and len(data_array.shape) >= 2 and data_array.shape[0] > 0:
                            # First column should be datetime (based on debug output)
                            try:
                                first_timestamp = data_array[0, 0]
                                last_timestamp = data_array[-1, 0]
                                
                                # Convert to datetime if they're pandas Timestamps
                                if hasattr(first_timestamp, 'to_pydatetime'):
                                    start_time = first_timestamp.to_pydatetime()
                                    end_time = last_timestamp.to_pydatetime()
                                elif hasattr(first_timestamp, 'timestamp'):
                                    start_time = datetime.fromtimestamp(first_timestamp.timestamp())
                                    end_time = datetime.fromtimestamp(last_timestamp.timestamp())
                                else:
                                    start_time = first_timestamp
                                    end_time = last_timestamp
                                
                                option_stats['start_time'] = start_time.isoformat()
                                option_stats['end_time'] = end_time.isoformat()
                                option_stats['duration_days'] = (end_time - start_time).days
                            except Exception as e:
                                logger.debug(f"Could not extract time info from {option_file}: {e}")
                    elif 'timestamps' in data.files:
                        timestamps = data['timestamps']
                        if len(timestamps) > 0:
                            start_time = datetime.fromtimestamp(timestamps[0])
                            end_time = datetime.fromtimestamp(timestamps[-1])
                            option_stats['start_time'] = start_time.isoformat()
                            option_stats['end_time'] = end_time.isoformat()
                            option_stats['duration_days'] = (end_time - start_time).days
                    
                    code_stats['options'][file_name] = option_stats
                    data.close()
                    
                except Exception as e:
                    logger.error(f"Error processing {option_file}: {e}")
                    code_stats['options'][option_file.stem] = {'error': str(e)}
            
            # Sort strikes and calculate range
            if code_stats['strikes']:
                code_stats['strikes'].sort()
                code_stats['strike_range'] = {
                    'min': min(code_stats['strikes']),
                    'max': max(code_stats['strikes'])
                }
            
            results['weekly_codes'][code] = code_stats
            
            # Accumulate totals
            total_options += code_stats['total_options']
            total_calls += code_stats['calls']
            total_puts += code_stats['puts']
            total_size += code_stats['file_size_mb']
        
        # Calculate summary statistics
        results['summary']['total_options'] = total_options
        results['summary']['total_calls'] = total_calls
        results['summary']['total_puts'] = total_puts
        results['summary']['total_file_size_mb'] = total_size
        
        if all_strikes:
            results['summary']['strike_range']['min'] = min(all_strikes)
            results['summary']['strike_range']['max'] = max(all_strikes)
        
        if results['total_weekly_codes'] > 0:
            results['summary']['avg_options_per_code'] = total_options / results['total_weekly_codes']
        
        self.options_stats = results
        logger.info(f"Found {total_options} options across {len(weekly_codes)} weekly codes")
        return results
    
    def scan_greeks_data(self) -> Dict[str, Any]:
        """
        Scan and analyze preprocessed Greeks data.
        
        Returns:
            Dict with Greeks data statistics
        """
        logger.info("Scanning preprocessed Greeks data...")
        
        if not self.greeks_path.exists():
            logger.warning(f"Greeks path does not exist: {self.greeks_path}")
            return {'error': 'Greeks path not found'}
        
        weekly_codes = [d for d in self.greeks_path.iterdir() if d.is_dir()]
        
        results = {
            'total_weekly_codes': len(weekly_codes),
            'weekly_codes': {},
            'summary': {
                'total_greeks_files': 0,
                'total_file_size_mb': 0,
                'avg_files_per_code': 0
            }
        }
        
        total_files = 0
        total_size = 0
        
        for code_dir in weekly_codes:
            code = code_dir.name
            greeks_files = list(code_dir.glob("*_greeks.npz"))
            
            code_stats = {
                'total_files': len(greeks_files),
                'file_size_mb': 0,
                'files': {}
            }
            
            for greeks_file in greeks_files:
                try:
                    file_size_mb = greeks_file.stat().st_size / (1024 * 1024)
                    code_stats['file_size_mb'] += file_size_mb
                    
                    # Load Greeks data for analysis
                    data = np.load(greeks_file, allow_pickle=True)
                    
                    file_stats = {
                        'file_size_mb': file_size_mb,
                        'fields_available': list(data.files)
                    }
                    
                    # Try to get data points
                    if 'delta' in data.files:
                        delta = data['delta']
                        file_stats['data_points'] = len(delta)
                    
                    # Check for timestamp information
                    if 'timestamps' in data.files:
                        timestamps = data['timestamps']
                        if len(timestamps) > 0:
                            start_time = datetime.fromtimestamp(timestamps[0])
                            end_time = datetime.fromtimestamp(timestamps[-1])
                            file_stats['start_time'] = start_time.isoformat()
                            file_stats['end_time'] = end_time.isoformat()
                            file_stats['duration_days'] = (end_time - start_time).days
                    
                    code_stats['files'][greeks_file.stem] = file_stats
                    data.close()
                    
                except Exception as e:
                    logger.error(f"Error processing {greeks_file}: {e}")
                    code_stats['files'][greeks_file.stem] = {'error': str(e)}
            
            results['weekly_codes'][code] = code_stats
            total_files += code_stats['total_files']
            total_size += code_stats['file_size_mb']
        
        # Calculate summary statistics
        results['summary']['total_greeks_files'] = total_files
        results['summary']['total_file_size_mb'] = total_size
        
        if results['total_weekly_codes'] > 0:
            results['summary']['avg_files_per_code'] = total_files / results['total_weekly_codes']
        
        self.greeks_stats = results
        logger.info(f"Found {total_files} Greeks files across {len(weekly_codes)} weekly codes")
        return results
    
    def generate_summary_statistics(self) -> Dict[str, Any]:
        """
        Generate overall summary statistics.
        
        Returns:
            Dict with summary statistics
        """
        logger.info("Generating summary statistics...")
        
        # Ensure all data is scanned
        if not self.underlying_stats:
            self.scan_underlying_data()
        if not self.options_stats:
            self.scan_options_data()
        if not self.greeks_stats:
            self.scan_greeks_data()
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'data_overview': {
                'underlying_assets': self.underlying_stats.get('total_files', 0),
                'weekly_option_codes': self.options_stats.get('total_weekly_codes', 0),
                'total_options': self.options_stats.get('summary', {}).get('total_options', 0),
                'preprocessed_greeks_codes': self.greeks_stats.get('total_weekly_codes', 0),
                'total_greeks_files': self.greeks_stats.get('summary', {}).get('total_greeks_files', 0)
            },
            'storage_analysis': {
                'underlying_data_mb': self.underlying_stats.get('summary', {}).get('total_file_size_mb', 0),
                'options_data_mb': self.options_stats.get('summary', {}).get('total_file_size_mb', 0),
                'greeks_data_mb': self.greeks_stats.get('summary', {}).get('total_file_size_mb', 0)
            }
        }
        
        # Calculate total storage
        summary['storage_analysis']['total_storage_mb'] = (
            summary['storage_analysis']['underlying_data_mb'] +
            summary['storage_analysis']['options_data_mb'] +
            summary['storage_analysis']['greeks_data_mb']
        )
        
        # Data completeness analysis
        completeness = {}
        if self.options_stats.get('weekly_codes'):
            greeks_codes = set(self.greeks_stats.get('weekly_codes', {}).keys())
            options_codes = set(self.options_stats.get('weekly_codes', {}).keys())
            
            completeness['options_with_greeks'] = len(greeks_codes & options_codes)
            completeness['options_without_greeks'] = len(options_codes - greeks_codes)
            completeness['greeks_coverage_percent'] = (
                completeness['options_with_greeks'] / len(options_codes) * 100
                if options_codes else 0
            )
        
        summary['data_completeness'] = completeness
        
        self.summary_stats = summary
        return summary
    
    def print_underlying_table(self):
        """Print formatted table of underlying data statistics."""
        if not self.underlying_stats:
            self.scan_underlying_data()
        
        print("\n" + "="*100)
        print("UNDERLYING DATA STATISTICS")
        print("="*100)
        
        files = self.underlying_stats.get('files', {})
        summary = self.underlying_stats.get('summary', {})
        
        print(f"Total Files: {self.underlying_stats.get('total_files', 0)}")
        print(f"Total Data Points: {summary.get('total_data_points', 0):,}")
        print(f"Total Storage: {summary.get('total_file_size_mb', 0):.2f} MB")
        print(f"Average Data Points per File: {summary.get('avg_data_points', 0):.0f}")
        
        if summary.get('date_range', {}).get('earliest'):
            print(f"Date Range: {summary['date_range']['earliest'][:10]} to {summary['date_range']['latest'][:10]}")
        
        print(f"\n{'Code':<8} | {'Points':<8} | {'Size(MB)':<8} | {'Days':<6} | {'Price Range':<18} | {'Returns':<15} | {'Volume':<12}")
        print("-" * 100)
        
        for code, stats in files.items():
            if 'error' in stats:
                print(f"{code:<8} | ERROR: {stats['error'][:70]}")
                continue
            
            points = stats.get('data_points', 0)
            size = stats.get('file_size_mb', 0)
            days = stats.get('duration_days', 0)
            
            price_stats = stats.get('price_stats', {})
            price_range = f"{price_stats.get('min', 0):.3f}-{price_stats.get('max', 0):.3f}" if price_stats else "N/A"
            
            returns_stats = stats.get('returns_stats', {})
            returns_info = f"{returns_stats.get('mean', 0):.6f}±{returns_stats.get('std', 0):.6f}" if returns_stats else "N/A"
            
            volume_stats = stats.get('volume_stats', {})
            volume_info = f"{volume_stats.get('total', 0):,}" if volume_stats else "N/A"
            
            print(f"{code:<8} | {points:<8,} | {size:<8.2f} | {days:<6} | {price_range:<18} | {returns_info:<15} | {volume_info:<12}")
    
    def print_options_table(self):
        """Print formatted table of options data statistics."""
        if not self.options_stats:
            self.scan_options_data()
        
        print("\n" + "="*100)
        print("OPTIONS DATA STATISTICS")
        print("="*100)
        
        summary = self.options_stats.get('summary', {})
        weekly_codes = self.options_stats.get('weekly_codes', {})
        
        print(f"Total Weekly Codes: {self.options_stats.get('total_weekly_codes', 0)}")
        print(f"Total Options: {summary.get('total_options', 0)} (Calls: {summary.get('total_calls', 0)}, Puts: {summary.get('total_puts', 0)})")
        print(f"Total Storage: {summary.get('total_file_size_mb', 0):.2f} MB")
        print(f"Strike Range: {summary.get('strike_range', {}).get('min', 0):.3f} - {summary.get('strike_range', {}).get('max', 0):.3f}")
        
        print(f"\n{'Code':<8} | {'Total':<6} | {'Calls':<6} | {'Puts':<6} | {'Size(MB)':<8} | {'Strike Range':<15} | {'Data Coverage':<12}")
        print("-" * 80)
        
        for code, stats in weekly_codes.items():
            total_options = stats.get('total_options', 0)
            calls = stats.get('calls', 0)
            puts = stats.get('puts', 0)
            size = stats.get('file_size_mb', 0)
            
            strike_range = stats.get('strike_range', {})
            if strike_range:
                strike_str = f"{strike_range.get('min', 0):.2f}-{strike_range.get('max', 0):.2f}"
            else:
                strike_str = "N/A"
            
            # Calculate data coverage (options with data points > 0)
            options = stats.get('options', {})
            with_data = sum(1 for opt in options.values() if isinstance(opt, dict) and opt.get('data_points', 0) > 0)
            coverage = f"{with_data}/{total_options}" if total_options > 0 else "N/A"
            
            print(f"{code:<8} | {total_options:<6} | {calls:<6} | {puts:<6} | {size:<8.2f} | {strike_str:<15} | {coverage:<12}")
    
    def print_greeks_table(self):
        """Print formatted table of Greeks data statistics."""
        if not self.greeks_stats:
            self.scan_greeks_data()
        
        print("\n" + "="*100)
        print("PREPROCESSED GREEKS STATISTICS")
        print("="*100)
        
        if 'error' in self.greeks_stats:
            print(f"ERROR: {self.greeks_stats['error']}")
            return
        
        summary = self.greeks_stats.get('summary', {})
        weekly_codes = self.greeks_stats.get('weekly_codes', {})
        
        print(f"Total Weekly Codes: {self.greeks_stats.get('total_weekly_codes', 0)}")
        print(f"Total Greeks Files: {summary.get('total_greeks_files', 0)}")
        print(f"Total Storage: {summary.get('total_file_size_mb', 0):.2f} MB")
        
        print(f"\n{'Code':<8} | {'Files':<6} | {'Size(MB)':<8} | {'Sample Coverage':<15}")
        print("-" * 50)
        
        for code, stats in weekly_codes.items():
            files_count = stats.get('total_files', 0)
            size = stats.get('file_size_mb', 0)
            
            # Sample a few files to show data coverage
            files = stats.get('files', {})
            sample_files = list(files.keys())[:3]
            sample_info = ", ".join([f.replace('_greeks', '') for f in sample_files])
            if len(files) > 3:
                sample_info += f", ...+{len(files)-3}"
            
            print(f"{code:<8} | {files_count:<6} | {size:<8.2f} | {sample_info[:15]:<15}")
    
    def print_summary_table(self):
        """Print overall summary table."""
        if not self.summary_stats:
            self.generate_summary_statistics()
        
        print("\n" + "="*80)
        print("DATA QUALITY SUMMARY")
        print("="*80)
        
        overview = self.summary_stats.get('data_overview', {})
        storage = self.summary_stats.get('storage_analysis', {})
        completeness = self.summary_stats.get('data_completeness', {})
        
        print("DATA OVERVIEW:")
        print(f"  Underlying Assets: {overview.get('underlying_assets', 0)}")
        print(f"  Weekly Option Codes: {overview.get('weekly_option_codes', 0)}")
        print(f"  Total Options: {overview.get('total_options', 0)}")
        print(f"  Greeks Files: {overview.get('total_greeks_files', 0)}")
        
        print(f"\nSTORAGE ANALYSIS:")
        print(f"  Underlying Data: {storage.get('underlying_data_mb', 0):.2f} MB")
        print(f"  Options Data: {storage.get('options_data_mb', 0):.2f} MB")
        print(f"  Greeks Data: {storage.get('greeks_data_mb', 0):.2f} MB")
        print(f"  Total Storage: {storage.get('total_storage_mb', 0):.2f} MB")
        
        if completeness:
            print(f"\nDATA COMPLETENESS:")
            print(f"  Options with Greeks: {completeness.get('options_with_greeks', 0)}")
            print(f"  Options without Greeks: {completeness.get('options_without_greeks', 0)}")
            print(f"  Greeks Coverage: {completeness.get('greeks_coverage_percent', 0):.1f}%")
        
        print(f"\nGenerated: {self.summary_stats.get('timestamp', 'Unknown')}")
    
    def export_to_csv(self, output_dir: str = "csv_process/data_quality_reports"):
        """
        Export all statistics to CSV files.
        
        Args:
            output_dir: Directory to save CSV files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export underlying statistics
        if self.underlying_stats:
            underlying_data = []
            for code, stats in self.underlying_stats.get('files', {}).items():
                if 'error' not in stats:
                    row = {
                        'code': code,
                        'data_points': stats.get('data_points', 0),
                        'file_size_mb': stats.get('file_size_mb', 0),
                        'duration_days': stats.get('duration_days', 0),
                        'start_time': stats.get('start_time', ''),
                        'end_time': stats.get('end_time', ''),
                        'price_min': stats.get('price_stats', {}).get('min', 0),
                        'price_max': stats.get('price_stats', {}).get('max', 0),
                        'price_mean': stats.get('price_stats', {}).get('mean', 0),
                        'price_std': stats.get('price_stats', {}).get('std', 0),
                        'volume_total': stats.get('volume_stats', {}).get('total', 0),
                        'returns_mean': stats.get('returns_stats', {}).get('mean', 0),
                        'returns_std': stats.get('returns_stats', {}).get('std', 0),
                        'sharpe_ratio': stats.get('returns_stats', {}).get('sharpe', 0)
                    }
                    underlying_data.append(row)
            
            if underlying_data:
                df = pd.DataFrame(underlying_data)
                df.to_csv(output_path / f"underlying_statistics_{timestamp}.csv", index=False)
        
        # Export options statistics
        if self.options_stats:
            options_data = []
            for code, stats in self.options_stats.get('weekly_codes', {}).items():
                for option_name, option_stats in stats.get('options', {}).items():
                    if 'error' not in option_stats:
                        row = {
                            'weekly_code': code,
                            'option_name': option_name,
                            'option_type': option_stats.get('type', ''),
                            'strike': option_stats.get('strike', 0),
                            'data_points': option_stats.get('data_points', 0),
                            'file_size_mb': option_stats.get('file_size_mb', 0),
                            'duration_days': option_stats.get('duration_days', 0),
                            'start_time': option_stats.get('start_time', ''),
                            'end_time': option_stats.get('end_time', '')
                        }
                        options_data.append(row)
            
            if options_data:
                df = pd.DataFrame(options_data)
                df.to_csv(output_path / f"options_statistics_{timestamp}.csv", index=False)
        
        # Export summary statistics
        if self.summary_stats:
            summary_data = [{
                'metric': 'underlying_assets',
                'value': self.summary_stats.get('data_overview', {}).get('underlying_assets', 0)
            }, {
                'metric': 'total_options',
                'value': self.summary_stats.get('data_overview', {}).get('total_options', 0)
            }, {
                'metric': 'total_storage_mb',
                'value': self.summary_stats.get('storage_analysis', {}).get('total_storage_mb', 0)
            }, {
                'metric': 'greeks_coverage_percent',
                'value': self.summary_stats.get('data_completeness', {}).get('greeks_coverage_percent', 0)
            }]
            
            df = pd.DataFrame(summary_data)
            df.to_csv(output_path / f"summary_statistics_{timestamp}.csv", index=False)
        
        logger.info(f"Statistics exported to {output_path}")
    
    def run_full_analysis(self, export_csv: bool = True):
        """
        Run complete data quality analysis and display all results.
        
        Args:
            export_csv: Whether to export results to CSV files
        """
        logger.info("Starting full data quality analysis...")
        
        # Scan all data sources
        self.scan_underlying_data()
        self.scan_options_data()
        self.scan_greeks_data()
        self.generate_summary_statistics()
        
        # Display all tables
        self.print_summary_table()
        self.print_underlying_table()
        self.print_options_table()
        self.print_greeks_table()
        
        # Export to CSV if requested
        if export_csv:
            self.export_to_csv()
        
        logger.info("Data quality analysis completed")


def main():
    """Main function to run data quality analysis."""
    analyzer = DataQualityStatistics()
    analyzer.run_full_analysis(export_csv=True)


if __name__ == "__main__":
    main()