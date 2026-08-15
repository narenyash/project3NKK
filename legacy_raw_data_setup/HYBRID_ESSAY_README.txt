# Hybrid Essay Generator with Local Ollama

## Setup Instructions

### Step 1: Install Ollama
1. Download Ollama from: https://ollama.ai
2. Install it on your system
3. Run Ollama service: `ollama serve` (in a terminal)

### Step 2: Pull a Model
In another terminal, pull a model:
```
ollama pull llama2
```

Other model options:
- `ollama pull neural-chat` - Faster, good for general tasks
- `ollama pull mistral` - Powerful, good quality
- `ollama pull dolphin-mixtral` - Advanced model

### Step 3: Run the Hybrid Essay Generator
```
python create_hybrid_essays.py
```

## What This Script Does

1. **Reads** all essay files from `class H` folder (Tab_1.txt to Tab_29.txt)
2. **Connects** to your local Ollama installation
3. **Generates** hybrid essays by:
   - Keeping ~60% of original text intact
   - Enhancing ~40% with AI-generated content
   - Improving transitions and flow
   - Maintaining original voice and authenticity
4. **Saves** hybrid versions as `Tab_1_hybrid.txt`, `Tab_2_hybrid.txt`, etc.

## Output Location
All hybrid essays will be saved in: `c:\naren yashwanth N\class H\`

## Features

- **Preserves Authenticity**: Keeps personal anecdotes, experiences, and core ideas
- **Maintains Structure**: Keeps the essay's overall organization and flow
- **Smart Enhancement**: Adds depth, improves vocabulary, better transitions
- **Batch Processing**: Processes all 29 essays automatically
- **Error Handling**: Falls back to original text if any error occurs

## Tips

- The first model in your Ollama list will be used automatically
- Larger models produce higher quality but take longer
- Processing each essay takes 30-60 seconds depending on model and system
- Each essay can be reviewed and edited manually after generation

## Troubleshooting

**Issue**: "Could not connect to Ollama"
- Solution: Make sure Ollama is running: `ollama serve`

**Issue**: "No models found in Ollama"
- Solution: Pull a model: `ollama pull llama2`

**Issue**: Script is very slow
- Solution: Try a faster model like neural-chat: `ollama pull neural-chat`

## File Structure After Generation

```
class H/
├── Tab_1.txt
├── Tab_1_hybrid.txt
├── Tab_2.txt
├── Tab_2_hybrid.txt
├── Tab_3.txt
├── Tab_3_hybrid.txt
...and so on
```
