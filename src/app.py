import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
from model import QuickDrawCNN
import torchvision.transforms as transforms
from PIL import Image

# Constants
MODEL_PATH = 'models/quickdraw_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Model
print("Loading model...")
try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False) # weights_only=False needed for dict
    class_names = checkpoint['class_names']
    
    model = QuickDrawCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state'])
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model (might be training still): {e}")
    model = None
    class_names = []

def predict(image):
    if model is None:
        return "Model not loaded correctly. Please ensure training is finished."
    
    if image is None:
        return "Please draw something."

    # Image processing
    # Gradio Sketchpad (numpy) -> (H, W, 3) usually 255=white, 0=black? Or depends on theme.
    # Usually white canvas, black ink.
    # We need: 28x28, 1 channel.
    # Training data: 0=bg, 255=stroke?
    
    # If image is a dictionary (common in newer Gradio versions for Sketchpad), extract composite
    if isinstance(image, dict):
        image = image['composite']

    # Let's convert to PIL
    img_pil = Image.fromarray(image.astype('uint8')).convert('L') # Grayscale
    
    # Resize to 28x28
    img_pil = img_pil.resize((28, 28))
    
    # Convert to numpy
    img_arr = np.array(img_pil)
    
    # Invert if the background is white (255) and stroke is black (0)
    # Quick Check: if mean is high (>127), it's likely white background.
    if img_arr.mean() > 127:
        img_arr = 255 - img_arr
        
    # Scale to 0-1
    img_arr = img_arr.astype(np.float32) / 255.0
    
    # Add batch and channel dims: (1, 1, 28, 28)
    tensor_input = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(tensor_input)
        # Softmax for probabilities
        probs = F.softmax(output, dim=1)
        
        # Get top 3
        top3_prob, top3_idx = torch.topk(probs, 3)
        
    results = {}
    for i in range(3):
        idx = top3_idx[0][i].item()
        prob = top3_prob[0][i].item()
        results[class_names[idx]] = prob
        
    return results

# Gradio Interface
# Gradio Interface
with gr.Blocks(title="Quick Draw Classifier", theme=gr.themes.Base()) as demo:
    gr.Markdown("# 🎨 Quick Draw Classifier")
    gr.Markdown("Draw one of the 345 Quick Draw categories and let the AI guess!")
    
    with gr.Tabs():
        # Tab 1: Main Drawing Area
        with gr.TabItem("✏️ Draw & Predict"):
            with gr.Row():
                with gr.Column(scale=2):
                    sketchpad = gr.Sketchpad(type="numpy", label="Draw Here", brush=gr.Brush(colors=["#000000"], color_mode="fixed"), height=400)
                    btn = gr.Button("🔮 Predict", variant="primary")
                
                with gr.Column(scale=1):
                    label = gr.Label(num_top_classes=3, label="Predictions")
            
            # Examples
            gr.Markdown("### 💡 Try these examples:")
            gr.Examples(
                examples="examples",
                inputs=sketchpad,
                label="Click an image to load it",
                examples_per_page=5
            )

            btn.click(fn=predict, inputs=sketchpad, outputs=label)
        
        # Tab 2: Categories Reference
        with gr.TabItem("📚 All Categories"):
            gr.Markdown("### Available Categories")
            gr.Markdown("The model can recognize the following objects:")
            gr.Markdown(", ".join(class_names) if len(class_names) > 0 else "Model loading...")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
