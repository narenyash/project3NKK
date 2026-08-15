import ollama
import json

print("Testing Ollama connection and model detection...\n")

try:
    print("Calling ollama.list()...")
    result = ollama.list()
    
    print(f"Result type: {type(result)}")
    print(f"Result: {result}")
    print(f"\nRaw JSON:")
    print(json.dumps(result, indent=2, default=str))
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
