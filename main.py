import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

st.set_page_config(page_title="Image_Generater",page_icon="📷",layout="wide")

st.title("📷Image Generater App")
# Load the model (CPU Optimized)
@st.cache_resource()
def load_model():
    model_id = "dreamlike-art/dreamlike-diffusion-1.0"
    
    # Load model with CPU optimization
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe.to("cpu")  # Force CPU usage
    return pipe

pipe = load_model()

# Function to generate an image
def generate_image(prompt, negative_prompt, num_steps):
    image = pipe(prompt, 
                 negative_prompt=negative_prompt, 
                 num_inference_steps=num_steps).images[0]
    return image

# Streamlit UI
st.title("🎨 Dreamlike Diffusion Image Generator (CPU Optimized)")

prompt = st.text_area("Enter your prompt:", "A futuristic city at sunset")
negative_prompt = st.text_area("Negative prompt (optional):", "blurry, low-quality, distorted")
num_steps = st.slider("Number of inference steps:", 5, 50, 20)  # Reduced default steps for CPU

if st.button("Generate Image"):
    with st.spinner("Generating... Please wait ⏳ (This may take some time..)"):
        image = generate_image(prompt, negative_prompt, num_steps)
        st.image(image, caption="Generated Image", use_column_width=True)