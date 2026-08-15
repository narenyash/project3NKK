import os
import re

# Define paths
input_file = r"c:\naren yashwanth N\OPEN_SOURCE_ESSAYS.txt"
output_folder = r"c:\naren yashwanth N\class H"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

print("\n" + "="*60)
print("SPLITTING OPEN SOURCE ESSAYS (Starting from Tab 29)")
print("="*60 + "\n")

# Read the input file
with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Split by "Tab X" pattern
tabs = re.split(r'(?=Tab \d+)', content)

# Track the starting number
current_tab_num = 29
created_count = 0

# Process each tab
for tab in tabs:
    tab_clean = tab.strip()
    if tab_clean:  # Skip empty tabs
        # Extract original tab number from the first line
        match = re.match(r'Tab (\d+)', tab_clean)
        if match:
            # Create filename with new numbering (starting from 29)
            output_filename = f"Tab_{current_tab_num}.txt"
            output_path = os.path.join(output_folder, output_filename)
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(tab_clean)
            
            print(f"✓ Created: {output_filename} (from original Tab {match.group(1)})")
            created_count += 1
            current_tab_num += 1

print(f"\n{'='*60}")
print(f"SPLITTING COMPLETE!")
print(f"{'='*60}")
print(f"✓ Successfully created: {created_count} essay files")
print(f"✓ Files saved in: {output_folder}")
print(f"✓ Numbering: Tab_29.txt to Tab_{current_tab_num - 1}.txt")
