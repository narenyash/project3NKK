import os
import re

try:
    import ollama
except ImportError:
    print("Installing ollama library...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'ollama'])
    import ollama

# Define paths
input_folder = r"c:\naren yashwanth N\class H"
output_folder = r"c:\naren yashwanth N\class AH"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

def create_hybrid_essay(original_text, tab_number, model_name):
    """
    Create a hybrid essay by mixing original content with AI-generated content.
    Keeps major parts intact while enhancing and expanding others.
    """
    
    # Split the essay into paragraphs
    paragraphs = [p.strip() for p in original_text.split('\n\n') if p.strip()]
    
    if not paragraphs:
        return original_text
    
    # Create a prompt for Ollama
    prompt = f"""I have a college essay with multiple paragraphs. I want you to create a hybrid version where:
1. Keep the main structure and core ideas exactly the same
2. Keep about 60% of the original text unchanged
3. Enhance and expand about 40% of the content with creative writing
4. Improve transitions between ideas
5. Add depth and detail to some sections while keeping the original voice

Original Essay (Tab {tab_number}):
---
{original_text}
---

Please provide the hybrid essay that maintains the original's authenticity while adding AI enhancement. 
Keep the essay about the same length. Preserve all personal anecdotes and experiences.
Make it flow better and sound more polished."""

    try:
        # Call Ollama with the prompt
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            stream=False
        )
        
        hybrid_essay = response['response'].strip()
        return hybrid_essay
    
    except Exception as e:
        print(f"  ⚠ Error with Ollama for Tab {tab_number}: {e}")
        print(f"  → Returning original text for Tab {tab_number}")
        return original_text

def process_essays(model_name):
    """Process only new essays (Tab_29 onwards) and create hybrid versions."""
    
    # Get all Tab_*.txt files starting from Tab_29
    essay_files = sorted(
        [f for f in os.listdir(input_folder) if f.startswith('Tab_') and f.endswith('.txt') and '_hybrid' not in f],
        key=lambda x: int(re.search(r'(\d+)', x).group(1))
    )
    
    # Filter to only process Tab_29 onwards
    essay_files = [f for f in essay_files if int(re.search(r'(\d+)', f).group(1)) >= 29]
    
    if not essay_files:
        print("No new essay files found (Tab_29 onwards)!")
        return
    
    print(f"Found {len(essay_files)} new essays to process (Tab_29 to Tab_97)\n")
    print("Starting hybrid essay generation...\n")
    
    successful = 0
    failed = 0
    
    for essay_file in essay_files:
        try:
            # Extract tab number
            match = re.search(r'Tab_(\d+)', essay_file)
            tab_number = match.group(1) if match else "unknown"
            
            # Read original essay
            input_path = os.path.join(input_folder, essay_file)
            with open(input_path, 'r', encoding='utf-8') as f:
                original_text = f.read()
            
            print(f"Processing: {essay_file}...", end=" ", flush=True)
            
            # Create hybrid version
            hybrid_text = create_hybrid_essay(original_text, tab_number, model_name)
            
            # Save hybrid version with _hybrid suffix
            output_filename = f"Tab_{tab_number}_hybrid.txt"
            output_path = os.path.join(output_folder, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(hybrid_text)
            
            print(f"✓ Saved as {output_filename}")
            successful += 1
            
        except Exception as e:
            print(f"✗ Error: {e}")
            failed += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"HYBRID ESSAY GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"✓ Successfully created: {successful} hybrid essays")
    if failed > 0:
        print(f"⚠ Failed: {failed} essays")
    print(f"\nHybrid essays saved to: {output_folder}")
    print(f"Files follow the pattern: Tab_X_hybrid.txt (where X >= 29)")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("HYBRID ESSAY GENERATOR FOR OPEN SOURCE ESSAYS")
    print("(Processing Tab_29 to Tab_97)")
    print("="*60 + "\n")
    
    # Check if Ollama is running
    try:
        available_models = ollama.list()
        
        # Extract model names from the response
        models_list = []
        if hasattr(available_models, 'models'):
            for m in available_models.models:
                if hasattr(m, 'model'):
                    models_list.append(m.model)
        
        print("✓ Ollama is running locally")
        
        if not models_list:
            print("✗ Error: No models found in Ollama")
            print("  Please pull a model first:")
            print("  Run this in a separate terminal window:")
            print("    python setup_ollama_model.py\n")
            exit(1)
        
        print(f"✓ Available models: {models_list}\n")
        
        # Use the first available model
        selected_model = models_list[0]
        print(f"✓ Using model: {selected_model}\n")
        
    except ConnectionError as e:
        print(f"✗ Error: Could not connect to Ollama")
        print(f"  Make sure Ollama is running on your system")
        print(f"  Open a terminal and run: ollama serve\n")
        exit(1)
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Process essays
    process_essays(selected_model)
