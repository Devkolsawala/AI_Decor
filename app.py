import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import requests
import json
import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

class WallDecorationAnalyzer:
    def __init__(self):
        """Initialize the analyzer with BLIP model and Ollama connection"""
        print("Loading BLIP model...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.ollama_url = "http://localhost:11434/api/generate"
        
    def extract_dominant_colors(self, image, n_colors=5):
        """Extract dominant colors from the image"""
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Reshape image to be a list of pixels
        pixels = img_rgb.reshape(-1, 3)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get colors and their percentages
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        counts = np.bincount(labels)
        percentages = counts / len(labels) * 100
        
        # Sort by percentage
        sorted_indices = np.argsort(percentages)[::-1]
        colors = colors[sorted_indices]
        percentages = percentages[sorted_indices]
        
        return colors, percentages
    
    def rgb_to_hex(self, rgb):
        """Convert RGB to hex color code"""
        return '#{:02x}{:02x}{:02x}'.format(int(rgb[2]), int(rgb[1]), int(rgb[0]))
    
    def get_color_name(self, rgb):
        """Get approximate color name"""
        r, g, b = int(rgb[2]), int(rgb[1]), int(rgb[0])
        
        if r > 200 and g > 200 and b > 200:
            return "White/Light"
        elif r < 50 and g < 50 and b < 50:
            return "Black/Dark"
        elif r > g and r > b:
            if r > 180:
                return "Red/Pink"
            else:
                return "Dark Red/Brown"
        elif g > r and g > b:
            return "Green"
        elif b > r and b > g:
            return "Blue"
        elif r > 150 and g > 150:
            return "Yellow"
        elif r > 100 and g < 100 and b < 100:
            return "Orange/Brown"
        elif r > 100 and b > 100:
            return "Purple/Magenta"
        else:
            return "Gray/Neutral"
    
    def analyze_image_with_blip(self, image):
        """Generate image caption using BLIP"""
        inputs = self.processor(image, return_tensors="pt")
        out = self.model.generate(**inputs, max_length=50)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    def get_suggestions_from_llama(self, wall_colors, image_description):
        """Get decoration suggestions from Llama via Ollama"""
        # Prepare color information
        color_info = "\n".join([
            f"- {self.get_color_name(color)} ({self.rgb_to_hex(color)}): {percentage:.1f}%"
            for color, percentage in wall_colors
        ])
        
        prompt = f"""You are an expert interior decorator. Analyze this wall and provide decoration suggestions.

Wall Description: {image_description}

Dominant Wall Colors:
{color_info}

Please provide:
1. **Balloon Color Suggestions**: Recommend 3-4 balloon colors that would complement this wall beautifully. Explain why each color works.
2. **Decoration Theme**: Suggest an overall theme based on the wall colors.
3. **Additional Decorations**: Recommend other decorative elements (streamers, banners, etc.) with colors.
4. **Arrangement Tips**: Brief tips on how to arrange the balloons.

Keep your response concise, practical, and creative."""

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": "llama3.2:3b",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                return f"Error: Unable to get suggestions. Status code: {response.status_code}"
        except Exception as e:
            return f"Error connecting to Ollama: {str(e)}\n\nMake sure Ollama is running with: ollama serve"
    
    def create_color_palette_image(self, colors, percentages, width=500, height=100):
        """Create a visual representation of the color palette"""
        palette = np.zeros((height, width, 3), dtype=np.uint8)
        
        start_x = 0
        for color, percentage in zip(colors, percentages):
            end_x = start_x + int(width * (percentage / 100))
            palette[:, start_x:end_x] = color
            start_x = end_x
        
        return Image.fromarray(cv2.cvtColor(palette, cv2.COLOR_BGR2RGB))
    
    def analyze(self, image):
        """Main analysis function"""
        if image is None:
            return "Please upload an image", None, "No analysis available"
        
        # Convert to PIL Image if necessary
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Analyze image with BLIP
        print("Analyzing image with BLIP...")
        description = self.analyze_image_with_blip(image)
        
        # Extract dominant colors
        print("Extracting dominant colors...")
        colors, percentages = self.extract_dominant_colors(image, n_colors=5)
        
        # Create color palette visualization
        palette_image = self.create_color_palette_image(colors, percentages)
        
        # Prepare wall color information
        wall_colors = list(zip(colors, percentages))
        
        # Get suggestions from Llama
        print("Getting decoration suggestions...")
        suggestions = self.get_suggestions_from_llama(wall_colors, description)
        
        # Format color information
        color_text = "**Detected Wall Colors:**\n\n"
        for color, percentage in wall_colors:
            color_name = self.get_color_name(color)
            hex_code = self.rgb_to_hex(color)
            color_text += f"🎨 {color_name} ({hex_code}): {percentage:.1f}%\n"
        
        return color_text, palette_image, suggestions

# Initialize analyzer
analyzer = WallDecorationAnalyzer()

# Create Gradio interface
def process_image(image):
    return analyzer.analyze(image)

# Build the interface
with gr.Blocks(title="Wall Decoration Suggester", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎈 Wall Decoration Suggestion System
    Upload a photo of your wall and get AI-powered suggestions for balloon colors and decorations!
    """)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Wall Image")
            analyze_btn = gr.Button("🎨 Analyze & Get Suggestions", variant="primary", size="lg")
        
        with gr.Column():
            color_output = gr.Markdown(label="Detected Colors")
            palette_output = gr.Image(label="Color Palette")
    
    suggestions_output = gr.Markdown(label="Decoration Suggestions")
    
    analyze_btn.click(
        fn=process_image,
        inputs=image_input,
        outputs=[color_output, palette_output, suggestions_output]
    )
    
    gr.Markdown("""
    ### 💡 Tips:
    - Take a well-lit photo of your wall
    - Ensure the wall is clearly visible
    - Avoid heavy shadows or reflections
    - Make sure Ollama is running in the background
    """)

if __name__ == "__main__":
    print("Starting Wall Decoration Suggestion System...")
    print("Make sure Ollama is running with: ollama serve")
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)