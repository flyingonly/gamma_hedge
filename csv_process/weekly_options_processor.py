import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import openpyxl
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WeeklyOptionsProcessor:
    """Weekly Options Data Processor
    
    Used for processing monthly Weekly Options Excel files to extract mapping information and options data
    """
    
    def __init__(self, data_dir: str = "data", output_dir: str = "processed_data"):
        """
        Initialize processor
        
        Args:
            data_dir: Original Excel files directory
            output_dir: Output directory
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Month mapping (from Weekly3 table month column to month numbers)
        self.month_mapping = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        
        # Weekly3 mapping table storage
        self.weekly_mapping = {}
        
    def calculate_third_friday(self, year: int, month: int) -> datetime:
        """
        Calculate the third Friday of the specified year and month
        
        Args:
            year: Year
            month: Month
            
        Returns:
            Date of the third Friday
        """
        # Find the first day of the month
        first_day = datetime(year, month, 1)
        
        # Find the first Friday
        days_until_friday = (4 - first_day.weekday()) % 7  # 0=Monday, 4=Friday
        first_friday = first_day + timedelta(days=days_until_friday)
        
        # Third Friday
        third_friday = first_friday + timedelta(weeks=2)
        
        # Ensure it's still in the same month
        if third_friday.month != month:
            # If it exceeds the month, the third Friday is in the next month, return the second Friday
            return first_friday + timedelta(weeks=1)
        
        return third_friday
    
    def load_weekly3_mapping(self, file_path: str, year: int = 2025) -> Dict:
        """
        Load Weekly3 mapping table and calculate expiry dates for each product
        
        Args:
            file_path: Excel file path
            year: Year, default 2025
            
        Returns:
            Dictionary containing mapping information
        """
        logger.info(f"Loading Weekly3 mapping table: {file_path}")
        
        try:
            # Read Weekly3 table, using header=0 to indicate the first row is column names
            df = pd.read_excel(file_path, sheet_name='Weekly3', engine='openpyxl', header=0)
            
            logger.info(f"Weekly3 table structure: {list(df.columns)}")
            logger.info(f"Data rows: {len(df)}")
            
            # Calculate expiry dates for each product
            products_info = []
            
            for _, row in df.iterrows():
                product_code = row['product']
                weekly_code = row['weekly_code']
                month_str = row['month']  # Jul, Aug, Sep etc
                month_code = row['month_code']  # N, Q, U etc
                spacing = row['spacing']
                underlying = row['underlying']
                
                # Calculate expiry date
                if month_str in self.month_mapping:
                    month_num = self.month_mapping[month_str]
                    expiry_date = self.calculate_third_friday(year, month_num)
                    
                    product_info = {
                        'product': product_code,
                        'weekly_code': weekly_code,
                        'month_code': month_code,
                        'month': month_str,
                        'month_num': month_num,
                        'spacing': spacing,
                        'underlying': underlying,
                        'expiry_date': expiry_date.strftime('%Y-%m-%d'),
                        'expiry_datetime': expiry_date,
                        'year': year
                    }
                    
                    products_info.append(product_info)
                    logger.info(f"Product {weekly_code}: {month_str} -> Expiry date {expiry_date.strftime('%Y-%m-%d')}")
                else:
                    logger.warning(f"Unrecognized month: {month_str}")
            
            mapping = {
                'file_path': file_path,
                'year': year,
                'products': products_info,
                'raw_data': df.to_dict('records')
            }
            
            return mapping
            
        except Exception as e:
            logger.error(f"Failed to load Weekly3 mapping table: {e}")
            raise
    
    def save_weekly_mapping(self, mapping_file: str = "weekly_mapping.json"):
        """
        Persist Weekly mapping table
        
        Args:
            mapping_file: Mapping file name
        """
        output_file = self.output_dir / mapping_file
        
        # Convert datetime objects to strings for JSON serialization
        serializable_mapping = {}
        for key, value in self.weekly_mapping.items():
            serializable_mapping[key] = {}
            for k, v in value.items():
                if k == 'products':
                    # Process products list
                    serializable_products = []
                    for product in v:
                        product_copy = product.copy()
                        if 'expiry_datetime' in product_copy:
                            del product_copy['expiry_datetime']
                        serializable_products.append(product_copy)
                    serializable_mapping[key][k] = serializable_products
                else:
                    serializable_mapping[key][k] = v
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_mapping, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Mapping table saved to: {output_file}")
    
    def load_weekly_mapping(self, mapping_file: str = "weekly_mapping.json") -> Dict:
        """
        Load saved Weekly mapping table
        
        Args:
            mapping_file: Mapping file name
            
        Returns:
            Mapping dictionary
        """
        mapping_path = self.output_dir / mapping_file
        
        if not mapping_path.exists():
            return {}
        
        with open(mapping_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_options_data(self, file_path: str, weekly_code: str, option_type: str) -> Optional[Dict]:
        """
        Extract options data, handle complex table structure
        
        Args:
            file_path: Excel file path
            weekly_code: Weekly code
            option_type: 'CALL' or 'PUT'
            
        Returns:
            Dictionary containing data for various strikes
        """
        sheet_name = f"{option_type}_{weekly_code}"
        
        try:
            # Read entire table as raw data
            df_raw = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl', header=None)
            
            logger.info(f"Read {sheet_name}, original size: {df_raw.shape}")
            
            # Analyze table structure
            strike_data = self.parse_options_sheet_structure(df_raw, weekly_code, option_type)
            
            return strike_data
            
        except Exception as e:
            logger.warning(f"Cannot read sheet {sheet_name}: {e}")
            return None
    
    def parse_options_sheet_structure(self, df_raw: pd.DataFrame, weekly_code: str, option_type: str) -> Dict:
        """
        Parse complex structure of options table
        
        Args:
            df_raw: Raw DataFrame
            weekly_code: Weekly code
            option_type: Option type
            
        Returns:
            Data dictionary organized by Strike
        """
        strike_data = {}
        
        # Find key rows
        # Row 5 (index 4) contains Strike prices
        # Row 7 (index 6) contains data column headers (DATETIME, LAST_PRICE, VOLUME)
        
        if len(df_raw) < 8:
            logger.warning(f"Insufficient data rows: {len(df_raw)}")
            return {}
        
        # Extract Strike information (Row 5, starting from column 4)
        strike_row = df_raw.iloc[4]  # Row 5, index 4
        strikes = []
        strike_positions = []
        
        for col_idx in range(4, len(strike_row)):
            cell_value = strike_row.iloc[col_idx]
            if pd.notna(cell_value) and str(cell_value).replace('.', '').replace('-', '').isdigit():
                try:
                    strike_price = float(cell_value)
                    strikes.append(strike_price)
                    strike_positions.append(col_idx)
                except:
                    continue
        
        logger.info(f"{option_type}_{weekly_code}: Found {len(strikes)} Strike prices")
        
        # Extract data for each Strike
        for i, strike in enumerate(strikes):
            col_start = strike_positions[i]
            
            # Each Strike occupies 4 columns: Ticker, DATETIME, LAST_PRICE, VOLUME
            # Based on observation, data starts from row 8
            strike_df_data = []
            
            # Extract data starting from row 8
            for row_idx in range(7, len(df_raw)):
                row_data = df_raw.iloc[row_idx]
                
                # Extract 4 columns of data for this Strike
                if col_start + 2 < len(row_data):  # Ensure sufficient columns
                    datetime_val = row_data.iloc[col_start]
                    price_val = row_data.iloc[col_start + 1] 
                    volume_val = row_data.iloc[col_start + 2]
                    
                    # Only keep valid data
                    if (pd.notna(datetime_val) and str(datetime_val) != '#N/A N/A' and 
                        pd.notna(price_val) and pd.notna(volume_val)):
                        
                        strike_df_data.append({
                            'datetime': datetime_val,
                            'last_price': price_val,
                            'volume': volume_val,
                            'strike': strike,
                            'weekly_code': weekly_code,
                            'option_type': option_type
                        })
            
            if strike_df_data:
                strike_df = pd.DataFrame(strike_df_data)
                strike_data[strike] = strike_df
                logger.debug(f"Strike {strike}: {len(strike_df_data)} valid data entries")
        
        return strike_data
    
    def get_weekly_codes_from_file(self, file_path: str) -> List[str]:
        """
        Extract Weekly code list from file
        
        Args:
            file_path: Excel file path
            
        Returns:
            Weekly code list
        """
        try:
            # Read all worksheet names
            xl_file = pd.ExcelFile(file_path, engine='openpyxl')
            sheet_names = xl_file.sheet_names
            
            # Extract codes from CALL and PUT sheets
            call_codes = [name.replace('CALL_', '') for name in sheet_names if name.startswith('CALL_')]
            put_codes = [name.replace('PUT_', '') for name in sheet_names if name.startswith('PUT_')]
            
            # Merge and deduplicate
            weekly_codes = list(set(call_codes + put_codes))
            
            logger.info(f"Extracted {len(weekly_codes)} Weekly codes from {Path(file_path).name}: {weekly_codes}")
            
            return weekly_codes
            
        except Exception as e:
            logger.error(f"Failed to extract Weekly codes: {e}")
            return []
    
    def process_single_file(self, file_path: str):
        """
        Process single Excel file
        
        Args:
            file_path: Excel file path
        """
        logger.info(f"Starting to process file: {file_path}")
        
        # 1. Load Weekly3 mapping
        mapping = self.load_weekly3_mapping(file_path)
        file_key = Path(file_path).stem
        self.weekly_mapping[file_key] = mapping
        
        # 2. Get Weekly codes
        weekly_codes = self.get_weekly_codes_from_file(file_path)
        
        # 3. Process each Weekly code
        for weekly_code in weekly_codes:
            self.process_weekly_code(file_path, weekly_code, mapping)
    
    def process_weekly_code(self, file_path: str, weekly_code: str, mapping: Dict):
        """
        Process data for single Weekly code
        
        Args:
            file_path: Excel file path
            weekly_code: Weekly code
            mapping: Mapping information
        """
        logger.info(f"Processing Weekly code: {weekly_code}")
        
        # Find information for this weekly_code from mapping
        product_info = None
        for product in mapping['products']:
            if product['weekly_code'] == weekly_code:
                product_info = product
                break
        
        if not product_info:
            logger.warning(f"Information for {weekly_code} not found in mapping table")
            return
        
        # Create Weekly code directory
        weekly_dir = self.output_dir / weekly_code
        weekly_dir.mkdir(exist_ok=True)
        
        # Extract CALL and PUT data
        for option_type in ['CALL', 'PUT']:
            strike_data_dict = self.extract_options_data(file_path, weekly_code, option_type)
            
            if not strike_data_dict:
                logger.warning(f"No data found for {option_type}_{weekly_code}")
                continue
            
            # Save data for each Strike
            for strike, strike_df in strike_data_dict.items():
                if len(strike_df) > 0:
                    # Add product information
                    strike_df = strike_df.copy()
                    strike_df['expiry_date'] = product_info['expiry_date']
                    strike_df['month'] = product_info['month']
                    strike_df['year'] = product_info['year']
                    strike_df['product'] = product_info['product']
                    strike_df['underlying'] = product_info['underlying']
                    
                    # Save as npz file
                    self.save_strike_data(strike_df, weekly_dir, option_type, strike)
    
    def save_strike_data(self, df: pd.DataFrame, weekly_dir: Path, option_type: str, strike: float):
        """
        Save Strike data to npz file
        
        Args:
            df: Strike data DataFrame
            weekly_dir: Weekly code directory
            option_type: 'CALL' or 'PUT'
            strike: Strike price
        """
        # Save as npz format
        filename = f"{option_type}_{strike}.npz"
        file_path = weekly_dir / filename
        
        # Convert DataFrame to numpy array for saving
        data_dict = {
            'data': df.values,
            'columns': df.columns.tolist(),
            'strike': strike,
            'option_type': option_type,
            'data_count': len(df)
        }
        
        np.savez_compressed(file_path, **data_dict)
        
        logger.debug(f"Saved {filename} ({len(df)} data entries) to {weekly_dir}")
    
    def process_all_files(self):
        """
        Process all Excel files
        """
        logger.info("Starting batch processing of all files")
        
        # Define file list
        files = [
            'Intraday_WeeklyOptions_Version_legacy_feb.xlsm',
            'Intraday_WeeklyOptions_Version_legacy_march.xlsm',
            'Intraday_WeeklyOptions_Version_legacy_april.xlsm',
            'Intraday_WeeklyOptions_Version_legacy_may.xlsm',
            'Intraday_WeeklyOptions_Version_legacy_june.xlsm',
            'Intraday_WeeklyOptions_Version_legacy_july.xlsm'
        ]
        
        processed_files = []
        failed_files = []
        
        for filename in files:
            file_path = self.data_dir / filename
            
            if not file_path.exists():
                logger.warning(f"File does not exist: {file_path}")
                failed_files.append(filename)
                continue
            
            try:
                self.process_single_file(str(file_path))
                processed_files.append(filename)
                logger.info(f"✓ Successfully processed: {filename}")
                
            except Exception as e:
                logger.error(f"✗ Processing failed {filename}: {e}")
                failed_files.append(filename)
        
        # Save mapping table
        self.save_weekly_mapping()
        
        # Generate processing report
        self.generate_report(processed_files, failed_files)
    
    def generate_report(self, processed_files: List[str], failed_files: List[str]):
        """
        Generate processing report
        
        Args:
            processed_files: List of successfully processed files
            failed_files: List of failed files
        """
        # Statistics for Weekly codes and data
        weekly_summary = {}
        
        for weekly_dir in self.output_dir.iterdir():
            if weekly_dir.is_dir() and not weekly_dir.name.startswith('.'):
                weekly_code = weekly_dir.name
                call_files = list(weekly_dir.glob("CALL_*.npz"))
                put_files = list(weekly_dir.glob("PUT_*.npz"))
                
                total_records = 0
                for npz_file in call_files + put_files:
                    try:
                        data = np.load(npz_file)
                        total_records += int(data.get('data_count', 0))
                    except:
                        continue
                
                weekly_summary[weekly_code] = {
                    'call_strikes': len(call_files),
                    'put_strikes': len(put_files),
                    'total_records': total_records
                }
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'processed_files': processed_files,
            'failed_files': failed_files,
            'total_files': len(processed_files) + len(failed_files),
            'success_rate': len(processed_files) / (len(processed_files) + len(failed_files)) * 100 if processed_files or failed_files else 0,
            'weekly_codes_count': len(weekly_summary),
            'weekly_summary': weekly_summary,
            'total_data_records': sum(info['total_records'] for info in weekly_summary.values())
        }
        
        # Save report
        report_file = self.output_dir / 'processing_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Processing report saved: {report_file}")
        logger.info(f"Processing completed: {len(processed_files)}/{len(processed_files) + len(failed_files)} files successful")
        logger.info(f"Processed {len(weekly_summary)} weekly codes, {report['total_data_records']} data records in total")
    
    def load_strike_data(self, weekly_code: str, option_type: str, strike: float) -> Optional[np.ndarray]:
        """
        Load data for specified strike
        
        Args:
            weekly_code: Weekly code
            option_type: 'CALL' or 'PUT'
            strike: Strike price
            
        Returns:
            numpy array data
        """
        file_path = self.output_dir / weekly_code / f"{option_type}_{strike}.npz"
        
        if not file_path.exists():
            logger.warning(f"File does not exist: {file_path}")
            return None
        
        try:
            data = np.load(file_path)
            return data
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return None
    
    def get_weekly_code_info(self, weekly_code: str) -> Optional[Dict]:
        """
        Get detailed information for specified weekly code
        
        Args:
            weekly_code: Weekly code
            
        Returns:
            Dictionary containing product information
        """
        for mapping in self.weekly_mapping.values():
            for product in mapping.get('products', []):
                if product['weekly_code'] == weekly_code:
                    return product
        return None


def main():
    """Main function example"""
    # Create processor instance
    processor = WeeklyOptionsProcessor(
        data_dir=".",  # Current directory, assuming Excel files are in current directory
        output_dir="weekly_options_data"
    )
    
    # Process all files
    processor.process_all_files()
    
    # Example: Load specific data
    # data = processor.load_strike_data('3CN5', 'CALL', 113.0)
    # if data is not None:
    #     print("Data shape:", data['data'].shape)
    #     print("Column names:", data['columns'])
    #     print("Data record count:", data['data_count'])


if __name__ == "__main__":
    main()
