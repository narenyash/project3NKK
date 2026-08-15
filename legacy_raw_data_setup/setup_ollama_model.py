import subprocess
import sys
import os

print("\n" + "="*60)
print("OLLAMA MODEL SETUP")
print("="*60 + "\n")

# List of available models to choose from
models = {
    "1": {"name": "llama2", "size": "3.8GB", "speed": "Fast", "quality": "Good"},
    "2": {"name": "neural-chat", "size": "4.1GB", "speed": "Very Fast", "quality": "Good"},
    "3": {"name": "mistral", "size": "4.1GB", "speed": "Fast", "quality": "Excellent"},
    "4": {"name": "dolphin-mixtral", "size": "26GB", "speed": "Slow", "quality": "Excellent"},
}

print("Available models:")
print("-" * 60)
for key, model in models.items():
    print(f"{key}. {model['name']:<20} Size: {model['size']:<8} Speed: {model['speed']:<12} Quality: {model['quality']}")

print("-" * 60)
print("Enter the number of the model you want to download:")
print("(Recommended: 1 or 2 for balanced speed/quality)\n")

choice = input("Choice (1-4): ").strip()

if choice not in models:
    print("Invalid choice!")
    sys.exit(1)

model_name = models[choice]["name"]

print(f"\nPulling model: {model_name}")
print("This may take a few minutes...\n")

try:
    result = subprocess.run(["ollama", "pull", model_name], capture_output=False)
    
    if result.returncode == 0:
        print(f"\n✓ Successfully downloaded {model_name}")
        print("\nYou can now run the hybrid essay generator:")
        print("  python create_hybrid_essays.py")
    else:
        print(f"\n✗ Error downloading {model_name}")
        print("Make sure Ollama is running (ollama serve in another terminal)")
        
except Exception as e:
    print(f"✗ Error: {e}")
    print("Make sure Ollama is installed and in your PATH")
