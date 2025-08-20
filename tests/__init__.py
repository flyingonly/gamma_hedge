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
    'delta_hedge_tests': True,  # Enable delta hedge tests if available
}

def check_delta_availability():
    """Check if delta hedge modules are available for testing"""
    try:
        import delta_hedge
        return True
    except ImportError:
        return False

# Global test settings
DELTA_AVAILABLE = check_delta_availability()
if not DELTA_AVAILABLE:
    print("⚠️  Delta hedge modules not available - some tests will be skipped")