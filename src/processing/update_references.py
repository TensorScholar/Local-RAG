#!/usr/bin/env python3
"""
Reference Alignment Utility for Advanced RAG System

This utility systematically updates component references in JavaScript files
to ensure consistency after renaming operations.

Author: Mohammad Atashi
Version: 1.0.0
"""

import os
import re
from pathlib import Path

def update_file_references(file_path):
    """
    Update component references within a JavaScript file.
    
    Args:
        file_path: Path to the JavaScript file
        
    Returns:
        int: Number of references updated
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Define patterns to match component imports with 'web-' prefix
    patterns = [
        (r'src="/assets/js/components/web-([^"]+)"', r'src="/assets/js/components/\1"'),
        (r'import\s+.*\s+from\s+[\'"]web-([^\'"]+)[\'"]', r'import from "\1"'),
        (r'web-assets/css/web-styles.css', r'assets/css/main.css'),
        (r'web-assets/js/components/web-', r'assets/js/components/'),
        (r'web-assets/js/utils/web-', r'assets/js/utils/')
    ]
    
    updated_content = content
    replace_count = 0
    
    for pattern, replacement in patterns:
        new_content, count = re.subn(pattern, replacement, updated_content)
        replace_count += count
        updated_content = new_content
    
    if replace_count > 0:
        with open(file_path, 'w') as f:
            f.write(updated_content)
        print(f"Updated {replace_count} references in {file_path}")
    
    return replace_count

def process_directory(directory):
    """
    Process all JavaScript files in a directory.
    
    Args:
        directory: Directory path to process
        
    Returns:
        int: Total number of references updated
    """
    total_updates = 0
    js_files = list(Path(directory).glob('**/*.js'))
    html_files = list(Path(directory).glob('**/*.html'))
    
    for file_path in js_files + html_files:
        total_updates += update_file_references(file_path)
    
    return total_updates

if __name__ == "__main__":
    web_dir = Path("web")
    if web_dir.exists():
        total = process_directory(web_dir)
        print(f"Total references updated: {total}")
    else:
        print("Web directory not found")
