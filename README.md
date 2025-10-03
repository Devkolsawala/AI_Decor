# Setup Guide for AI Decor

This guide will walk you through setting up the AI Decor application on your local machine.

## Prerequisites

- Python 3.8 or higher
- Git
- 16GB RAM (minimum 8GB)
- ~5GB free disk space

## Step-by-Step Installation

### 1. Install Python

If you don't have Python 3.8+:

**Windows:**
- Download from [python.org](https://www.python.org/downloads/)
- During installation, check "Add Python to PATH"

**Verify installation:**
```bash
python --version
```

### 2. Install Git

**Windows:**
- Download from [git-scm.com](https://git-scm.com/downloads)

**Verify installation:**
```bash
git --version
```

### 3. Clone the Repository

```bash
git clone https://github.com/Devkolsawala/AI_Decor.git
cd AI_Decor
```

### 4. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your command line.

### 5. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- PyTorch (CPU version)
- Transformers (Hugging Face)
- Gradio
- OpenCV
- scikit-learn
- And other dependencies

**Note:** First-time installation may take 5-10 minutes depending on your internet speed.

### 6. Install Ollama

**Windows:**
1. Download Ollama from [https://ollama.ai/download](https://ollama.ai/download)
2. Run the installer
3. Ollama will start automatically

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Mac:**
```bash
brew install ollama
```

### 7. Download AI Models

**Start Ollama server:**
```bash
ollama serve
```

**In a new terminal, pull the Llama model:**
```bash
ollama pull llama3.2:3b
```

This will download ~2GB of model data.

**Verify model is downloaded:**
```bash
ollama list
```

You should see `llama3.2:3b` in the list.

### 8. First Run

With Ollama running in the background:

```bash
python app.py
```

**Expected output:**
```
Loading BLIP model...
Starting Wall Decoration Suggestion System...
Make sure Ollama is running with: ollama serve
Running on local URL:  http://127.0.0.1:7860
```

### 9. Access the Application

Open your browser and go to:
```
http://127.0.0.1:7860
```

## Testing the Installation

1. Upload a test image of a wall (any photo with a visible wall)
2. Click "Analyze & Get Suggestions"
3. Wait 10-30 seconds
4. You should see:
   - Detected wall colors
   - Color palette visualization
   - AI-generated decoration suggestions

## Common Installation Issues

### Issue 1: "Python not found"
**Solution:** Make sure Python is added to PATH during installation, or reinstall Python with "Add to PATH" checked.

### Issue 2: "pip install fails"
**Solution:** 
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### Issue 3: "Ollama connection refused"
**Solution:** 
- Make sure Ollama is running: `ollama serve`
- Check if port 11434 is available
- Try restarting Ollama

### Issue 4: "CUDA out of memory"
**Solution:** This shouldn't happen as we use CPU version, but if it does:
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Issue 5: Models downloading slowly
**Solution:** 
- Be patient, first download takes time
- Check your internet connection
- Models are cached after first download

### Issue 6: "Port 7860 already in use"
**Solution:** Change the port in `app.py`:
```python
demo.launch(share=False, server_name="127.0.0.1", server_port=7861)
```

## Performance Optimization

### For Slower Systems:

**1. Use lighter Llama model:**
```bash
ollama pull llama3.2:1b
```

Update in `app.py`:
```python
"model": "llama3.2:1b"
```

**2. Reduce image processing:**
- Resize images before uploading (max 512x512)
- Reduce number of colors extracted in app.py

**3. Close unnecessary applications:**
- Free up RAM
- Close browser tabs
- Stop background processes

## Updating the Application

```bash
cd AI_Decor
git pull origin main
pip install -r requirements.txt --upgrade
```

## Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Remove project folder
cd ..
rm -rf AI_Decor  # Linux/Mac
# or
rmdir /s AI_Decor  # Windows

# Uninstall Ollama (optional)
# Windows: Use "Add or Remove Programs"
# Linux: Follow Ollama documentation
```

## Need Help?

- Check [README.md](README.md) for usage instructions
- Open an issue on [GitHub](https://github.com/Devkolsawala/AI_Decor/issues)
- Review [Troubleshooting section](README.md#-troubleshooting)

## Next Steps

After successful installation:
1. Read [README.md](README.md) for usage tips
2. Try with different wall images
3. Experiment with decoration suggestions
4. Consider contributing improvements

---

Happy decorating! 🎈
