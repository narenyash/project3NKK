import os
import shutil
from pathlib import Path

# Define paths
source_folder = r"c:\naren yashwanth N\class H"
dest_folder = r"c:\naren yashwanth N\class AH"

print("\n" + "="*60)
print("MOVING HYBRID ESSAYS TO 'class AH' FOLDER")
print("="*60 + "\n")

# Create destination folder if it doesn't exist
try:
    os.makedirs(dest_folder, exist_ok=True)
    print(f"✓ Destination folder created/verified: {dest_folder}\n")
except Exception as e:
    print(f"✗ Error creating folder: {e}")
    exit(1)

# Find all hybrid essay files
hybrid_files = []
try:
    for file in os.listdir(source_folder):
        if file.endswith('_hybrid.txt') and file.startswith('Tab_'):
            hybrid_files.append(file)
    
    hybrid_files.sort()
    print(f"Found {len(hybrid_files)} hybrid essay files to move:\n")
    
except Exception as e:
    print(f"✗ Error reading source folder: {e}")
    exit(1)

if not hybrid_files:
    print("✗ No hybrid essay files found!")
    exit(1)

# Move files
moved_count = 0
failed_count = 0

for file in hybrid_files:
    try:
        source_path = os.path.join(source_folder, file)
        dest_path = os.path.join(dest_folder, file)
        
        # Move the file
        shutil.move(source_path, dest_path)
        print(f"  ✓ {file}")
        moved_count += 1
        
    except Exception as e:
        print(f"  ✗ {file} - Error: {e}")
        failed_count += 1

# Print summary
print(f"\n{'='*60}")
print("MOVE COMPLETE!")
print(f"{'='*60}")
print(f"✓ Successfully moved: {moved_count} files")
if failed_count > 0:
    print(f"✗ Failed: {failed_count} files")
print(f"\nHybrid essays are now in: {dest_folder}")
