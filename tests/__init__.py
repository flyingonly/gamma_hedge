"""
Test suite for gamma hedge project
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_CONFIG = {
    'timeout': 300,  # 5 minutes
    'mock_data': True,
    'gpu_tests': False,  # Disable GPU tests by default
}