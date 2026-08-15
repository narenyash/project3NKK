import os
import re

# Define paths
input_file = r"C:\naren yashwanth N\OPEN_SOURCE_ESSAYS.txt"
output_folder = r"c:\naren yashwanth N\class AH"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Read the input file
with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Split by "Tab X" pattern
# The pattern looks for "Tab" followed by a number at the start of a line
tabs = re.split(r'(?=Tab \d+)', content)

# Process each tab
for tab in tabs:
    tab_clean = tab.strip()
    if tab_clean:  # Skip empty tabs
        # Extract tab number from the first line
        match = re.match(r'Tab (\d+)', tab_clean)
        if match:
            tab_number = match.group(1)
            # Create filename
            output_filename = f"Tab_{tab_number}.txt"
            output_path = os.path.join(output_folder, output_filename)
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(tab_clean)
            
            print(f"✓ Created: {output_filename}")

print(f"\n✓ All {len([t for t in tabs if t.strip()])} tabs have been separated and saved to '{output_folder}'")
