#!/usr/bin/env python3
"""
Configuration management command line tool
Provides utilities for managing gamma hedge configurations
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.config import (
    get_config, get_single_config, update_config, reload_config, 
    save_config, get_config_status, setup_config_system,
    enable_auto_reload, disable_auto_reload
)

def cmd_show(args):
    """Show current configuration"""
    if args.type:
        # Show specific config type
        try:
            config = get_single_config(args.type)
            config_dict = config.to_dict()
            print(f"Configuration '{args.type}':")
            print(json.dumps(config_dict, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Error: {e}")
            return 1
    else:
        # Show all configurations
        try:
            configs = get_config()
            print("All configurations:")
            for name, config in configs.items():
                print(f"\n=== {name} ===")
                config_dict = config.to_dict()
                print(json.dumps(config_dict, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Error: {e}")
            return 1
    
    return 0

def cmd_update(args):
    """Update configuration values"""
    try:
        # Parse updates from command line
        updates = {}
        for update in args.updates:
            if '=' not in update:
                print(f"Error: Invalid update format '{update}'. Use key=value")
                return 1
            
            key, value = update.split('=', 1)
            
            # Try to parse as JSON for complex values
            try:
                updates[key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                # Treat as string
                updates[key] = value
        
        # Apply updates
        update_config(args.type, updates, validate=not args.no_validate)
        print(f"Updated configuration '{args.type}' successfully")
        
        # Save if requested
        if args.save:
            save_config(args.type, args.source_index)
            print(f"Saved configuration to source {args.source_index}")
        
        return 0
        
    except Exception as e:
        print(f"Error updating configuration: {e}")
        return 1

def cmd_reload(args):
    """Reload configuration from sources"""
    try:
        reload_config()
        print("Configuration reloaded successfully")
        return 0
    except Exception as e:
        print(f"Error reloading configuration: {e}")
        return 1

def cmd_status(args):
    """Show configuration system status"""
    try:
        status = get_config_status()
        print("Configuration System Status:")
        print(f"  Registered types: {', '.join(status['registered_types'])}")
        print(f"  Loaded configs: {', '.join(status['loaded_configs'])}")
        print(f"  Auto-reload: {status['auto_reload']}")
        print(f"  Reload interval: {status['reload_interval']}s")
        print(f"  Sources count: {status['sources_count']}")
        return 0
    except Exception as e:
        print(f"Error getting status: {e}")
        return 1

def cmd_validate(args):
    """Validate configuration"""
    try:
        if args.type:
            # Validate specific config
            config = get_single_config(args.type)
            print(f"Configuration '{args.type}' is valid")
        else:
            # Validate all configs
            configs = get_config()
            for name in configs:
                get_single_config(name)  # This will validate
            print("All configurations are valid")
        return 0
    except Exception as e:
        print(f"Validation error: {e}")
        return 1

def cmd_auto_reload(args):
    """Control automatic configuration reloading"""
    try:
        if args.enable:
            enable_auto_reload(args.interval)
            print(f"Auto-reload enabled with {args.interval}s interval")
        else:
            disable_auto_reload()
            print("Auto-reload disabled")
        return 0
    except Exception as e:
        print(f"Error controlling auto-reload: {e}")
        return 1

def cmd_export(args):
    """Export configuration to file"""
    try:
        if args.type:
            config = get_single_config(args.type)
            export_data = {args.type: config.to_dict()}
        else:
            configs = get_config()
            export_data = {name: config.to_dict() for name, config in configs.items()}
        
        # Write to file
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"Configuration exported to {output_path}")
        return 0
        
    except Exception as e:
        print(f"Error exporting configuration: {e}")
        return 1

def cmd_import(args):
    """Import configuration from file"""
    try:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: File {input_path} does not exist")
            return 1
        
        # Load configuration from file
        with open(input_path, 'r', encoding='utf-8') as f:
            import_data = json.load(f)
        
        # Apply configurations
        for config_type, config_values in import_data.items():
            update_config(config_type, config_values, validate=not args.no_validate)
            print(f"Imported configuration '{config_type}'")
            
            # Save if requested
            if args.save:
                save_config(config_type, args.source_index)
        
        print("Configuration import completed")
        return 0
        
    except Exception as e:
        print(f"Error importing configuration: {e}")
        return 1

def main():
    """Main command line interface"""
    parser = argparse.ArgumentParser(description="Gamma Hedge Configuration Manager")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show current configuration')
    show_parser.add_argument('--type', '-t', help='Configuration type to show')
    show_parser.set_defaults(func=cmd_show)
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Update configuration values')
    update_parser.add_argument('type', help='Configuration type to update')
    update_parser.add_argument('updates', nargs='+', 
                              help='Updates in key=value format')
    update_parser.add_argument('--no-validate', action='store_true',
                              help='Skip validation')
    update_parser.add_argument('--save', action='store_true',
                              help='Save configuration after update')
    update_parser.add_argument('--source-index', type=int, default=1,
                              help='Source index to save to (default: 1)')
    update_parser.set_defaults(func=cmd_update)
    
    # Reload command
    reload_parser = subparsers.add_parser('reload', help='Reload configuration from sources')
    reload_parser.set_defaults(func=cmd_reload)
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show configuration system status')
    status_parser.set_defaults(func=cmd_status)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration')
    validate_parser.add_argument('--type', '-t', help='Configuration type to validate')
    validate_parser.set_defaults(func=cmd_validate)
    
    # Auto-reload command
    auto_reload_parser = subparsers.add_parser('auto-reload', help='Control automatic reloading')
    auto_reload_parser.add_argument('--enable', action='store_true', help='Enable auto-reload')
    auto_reload_parser.add_argument('--disable', dest='enable', action='store_false', 
                                   help='Disable auto-reload')
    auto_reload_parser.add_argument('--interval', type=float, default=5.0,
                                   help='Reload interval in seconds (default: 5.0)')
    auto_reload_parser.set_defaults(func=cmd_auto_reload)
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export configuration to file')
    export_parser.add_argument('output', help='Output file path')
    export_parser.add_argument('--type', '-t', help='Configuration type to export')
    export_parser.set_defaults(func=cmd_export)
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import configuration from file')
    import_parser.add_argument('input', help='Input file path')
    import_parser.add_argument('--no-validate', action='store_true',
                              help='Skip validation')
    import_parser.add_argument('--save', action='store_true',
                              help='Save configuration after import')
    import_parser.add_argument('--source-index', type=int, default=1,
                              help='Source index to save to (default: 1)')
    import_parser.set_defaults(func=cmd_import)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())