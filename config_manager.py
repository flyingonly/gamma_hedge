#!/usr/bin/env python3
"""
Configuration management command line tool
Provides utilities for managing gamma hedge configurations using the unified core.config system
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.config import create_config, Config
from dataclasses import asdict

def cmd_show(args):
    """Show current configuration"""
    try:
        config = create_config()
        
        if args.type:
            # Show specific config type
            if hasattr(config, args.type):
                config_obj = getattr(config, args.type)
                config_dict = asdict(config_obj)
                print(f"Configuration '{args.type}':")
                print(json.dumps(config_dict, indent=2, ensure_ascii=False))
            else:
                print(f"Error: Configuration type '{args.type}' not found.")
                print("Available types: market, training, model, delta_hedge, greeks")
                return 1
        else:
            # Show all configurations
            print("All configurations:")
            config_types = ['market', 'training', 'model', 'delta_hedge', 'greeks']
            for config_type in config_types:
                print(f"\n=== {config_type} ===")
                config_obj = getattr(config, config_type)
                config_dict = asdict(config_obj)
                print(json.dumps(config_dict, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

def cmd_status(args):
    """Show configuration system status"""
    try:
        config = create_config()
        print("Configuration System Status:")
        print("✅ Configuration system: Operational")
        print("✅ Configuration types: market, training, model, delta_hedge, greeks")
        print(f"✅ Source priority: CLI args → config files → environment → defaults")
        
        # Show basic info about each config type
        config_types = ['market', 'training', 'model', 'delta_hedge', 'greeks']
        for config_type in config_types:
            config_obj = getattr(config, config_type)
            num_fields = len([k for k in asdict(config_obj).keys() if not k.startswith('_')])
            print(f"  - {config_type}: {num_fields} configuration parameters")
        
        return 0
    except Exception as e:
        print(f"Error getting configuration status: {e}")
        return 1

def cmd_update(args):
    """Update configuration - NOT SUPPORTED"""
    print("Error: Configuration update is not supported in the current system.")
    print("To modify configurations, please:")
    print("1. Edit config/default.json or config/development.json")
    print("2. Set environment variables with GAMMA_HEDGE_ prefix")
    print("3. Use command line arguments when running applications")
    return 1

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Configuration management tool')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Show command
    parser_show = subparsers.add_parser('show', help='Show configuration')
    parser_show.add_argument('--type', help='Configuration type to show')
    
    # Status command  
    parser_status = subparsers.add_parser('status', help='Show system status')
    
    # Update command (disabled)
    parser_update = subparsers.add_parser('update', help='Update configuration (disabled)')
    parser_update.add_argument('type', help='Configuration type')
    parser_update.add_argument('updates', nargs='*', help='Key=value updates')
    
    args = parser.parse_args()
    
    if args.command == 'show':
        return cmd_show(args)
    elif args.command == 'status':
        return cmd_status(args)
    elif args.command == 'update':
        return cmd_update(args)
    else:
        parser.print_help()
        return 0

if __name__ == '__main__':
    sys.exit(main())