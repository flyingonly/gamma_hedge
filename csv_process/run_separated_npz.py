#!/usr/bin/env python3
"""
One-click run script - Process Excel file and generate separated NPZ format data
Each underlying code is saved as an independent NPZ file
"""

import sys
import os
from pathlib import Path

def check_dependencies():
    """Check required packages"""
    required_packages = ['pandas', 'numpy', 'openpyxl']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def find_excel_file():
    """Find Excel file"""
    possible_names = [
        'Underlying_Intraday_YTD.xlsm',
        'Underlying_Intraday_YTD.xlsx', 
        'underlying_data.xlsm',
        'underlying_data.xlsx'
    ]
    
    # Check current directory
    for name in possible_names:
        if os.path.exists(name):
            return name
    
    # Let user input
    print("Excel file not found, possible file names:")
    for i, name in enumerate(possible_names, 1):
        print(f"  {i}. {name}")
    
    user_input = input("\nPlease enter Excel file path: ").strip()
    if user_input and os.path.exists(user_input):
        return user_input
    
    return None

def main():
    """Main function"""
    print("🚀 Underlying Data Processor - Separated NPZ Format")
    print("="*70)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Import processor
    try:
        from separated_npz_processor import SeparatedNPZProcessor
        from separated_npz_loader import SeparatedNPZLoader
    except ImportError as e:
        print(f"❌ Failed to import modules: {e}")
        print("Please ensure separated_npz_processor.py and separated_npz_loader.py are in the same directory")
        return 1
    
    # Find Excel file
    excel_file = find_excel_file()
    if not excel_file:
        print("❌ Excel file not found, exiting program")
        return 1
    
    print(f"✅ Found Excel file: {excel_file}")
    
    # Set output directory
    output_dir = './underlying_npz'
    print(f"📁 Output directory: {output_dir}")
    
    try:
        # Create processor
        processor = SeparatedNPZProcessor(excel_file, output_dir)
        
        # Execute processing
        print("\nStarting data processing...")
        success = processor.process_all()
        
        if success:
            print("\n🎉 Data processing completed!")
            
            # Verify generated files
            print("\nVerifying NPZ files...")
            try:
                loader = SeparatedNPZLoader(output_dir)
                loader.print_summary()
                
                print(f"\n✅ Separated NPZ file verification successful")
                print(f"📂 Files saved to: {Path(output_dir).absolute()}")
                
                # File statistics
                npz_files = list(Path(output_dir).glob('*.npz'))
                data_files = [f for f in npz_files if f.name != 'index.npz']
                
                print(f"\n📊 File statistics:")
                print(f"  Data files: {len(data_files)}")
                print(f"  Index files: 1 (index.npz)")
                print(f"  Total files: {len(npz_files)}")
                
                # Display file list
                print(f"\n📋 Generated files:")
                for f in sorted(npz_files):
                    size_kb = f.stat().st_size / 1024
                    print(f"  {f.name:<15} ({size_kb:.1f} KB)")
                
                # Provide usage suggestions
                print("\n📋 Usage instructions:")
                print("1. Direct NPZ file usage:")
                print("   import numpy as np")
                print("   data = np.load('underlying_npz/USU5.npz')")
                print("   timestamps = data['timestamps']")
                print("   prices = data['prices']")
                print("   volumes = data['volumes']")
                print()
                print("2. Using data loader (recommended):")
                print("   from separated_npz_loader import SeparatedNPZLoader")
                print("   loader = SeparatedNPZLoader('./underlying_npz')")
                print("   timestamps, prices, volumes = loader.load_arrays('USU5')")
                print()
                print("3. Run analysis example:")
                print("   python separated_npz_loader.py")
                print()
                print("4. View generated example code:")
                print("   python underlying_npz/usage_example.py")
                print()
                print("5. Batch analysis:")
                print("   loader = SeparatedNPZLoader('./underlying_npz')")
                print("   loader.compare_underlyings()")
                print("   loader.calculate_correlation_matrix()")
                
                # Show individual file usage example
                codes = loader.get_codes()
                if codes:
                    example_code = codes[0]
                    print(f"\n💡 Quick start example (using {example_code}):")
                    print(f"   from separated_npz_loader import load_underlying")
                    print(f"   timestamps, prices, volumes = load_underlying('{example_code}')")
                    print(f"   print(f'Data points: {{len(prices):,}}')")
                    print(f"   print(f'Price range: {{prices.min():.5f}} - {{prices.max():.5f}}')")
                
            except Exception as e:
                print(f"⚠️  NPZ file verification failed: {e}")
                print("But data processing completed, files may still be usable")
        
        else:
            print("❌ Data processing failed")
            return 1
    
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())