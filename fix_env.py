#!/usr/bin/env python3
"""
Script to fix .env file - ensures OPENAI_API_KEY is on a single line
"""
import re
from pathlib import Path

def fix_env_file(env_path: Path):
    """Fix .env file by ensuring OPENAI_API_KEY is on a single line"""
    if not env_path.exists():
        print(f"❌ .env file not found at {env_path}")
        return False
    
    print(f"Reading .env file from {env_path}")
    with open(env_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all OPENAI_API_KEY entries
    lines = content.split('\n')
    new_lines = []
    i = 0
    api_key_found = False
    api_key_value = None
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this line starts with OPENAI_API_KEY
        if re.match(r'^\s*OPENAI_API_KEY\s*[=:]\s*(.+)', line, re.IGNORECASE):
            # Extract the key value (might be partial)
            match = re.match(r'^\s*OPENAI_API_KEY\s*[=:]\s*(.+)', line, re.IGNORECASE)
            if match:
                api_key_value = match.group(1).strip().strip('"\'')
                api_key_found = True
                
                # Check if the value continues on next lines
                i += 1
                while i < len(lines) and not re.match(r'^\s*[A-Z_]+\s*[=:]', lines[i]):
                    # This line is a continuation of the API key
                    next_line = lines[i].strip().strip('"\'')
                    if next_line and not next_line.startswith('#'):
                        api_key_value += next_line
                    i += 1
                
                # Write the cleaned API key on a single line
                # Remove any whitespace/newlines from the key
                api_key_value = ''.join(api_key_value.split())
                new_lines.append(f'OPENAI_API_KEY={api_key_value}')
                continue
        
        # Regular line - add it
        new_lines.append(line)
        i += 1
    
    # If we found and fixed the API key, write it back
    if api_key_found and api_key_value:
        # Remove any duplicate OPENAI_API_KEY entries
        final_lines = []
        seen_openai_key = False
        for line in new_lines:
            if re.match(r'^\s*OPENAI_API_KEY\s*[=:]', line, re.IGNORECASE):
                if not seen_openai_key:
                    final_lines.append(line)
                    seen_openai_key = True
            else:
                final_lines.append(line)
        
        # Write back to file
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(final_lines))
        
        print(f"✓ Fixed .env file - OPENAI_API_KEY is now on a single line")
        print(f"  Key length: {len(api_key_value)} characters")
        print(f"  Key preview: {api_key_value[:10]}...{api_key_value[-4:]}")
        return True
    else:
        print("⚠️  No OPENAI_API_KEY found in .env file")
        return False

if __name__ == "__main__":
    backend_dir = Path(__file__).parent
    env_file = backend_dir / ".env"
    
    if fix_env_file(env_file):
        print("\n✓ .env file has been fixed!")
        print("  Please restart your application for changes to take effect.")
    else:
        print("\n❌ Failed to fix .env file")

