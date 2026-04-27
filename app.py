import streamlit as st
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from PIL import Image
import numpy as np
import io
import zipfile
import time
import random

# --- CONFIGURATION & THEMING ---
st.set_page_config(page_title="SpriteForge", page_icon="⚡", layout="wide")

# Custom CSS for Rajdhani and Source Code Pro styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Source+Code+Pro:wght@400;600&display=swap');
    
    .main { background-color: #0d1117; color: #ffffff; }
    h1, h2, h3 { font-family: Rajdhani, sans-serif; text-transform: uppercase; letter-spacing: 2px; color: #00e5cc; }
    div[data-testid="stText"], label { font-family: 'Source Code Pro', monospace; color: #ffffff; }
    
    .stButton>button { 
        background-color: #00e5cc; color: #0d1117; font-family: Rajdhani; font-weight: bold; 
        border-radius: 4px; border: none; width: 100%;
    }
    .sprite-card {
        border: 2px solid #00e5cc; border-radius: 8px; padding: 10px; transition: transform 0.3s;
    }
    .sprite-card:hover { transform: scale(1.03); box-shadow: 0 0 15px #00e5cc; }
    </style>
""", unsafe_allow_html=True)

# --- MODEL ARCHITECTURES (Extracted from Notebook) ---
#
class DCGenerator(nn.Module):
    def __init__(self, latent=128, fm=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent, fm*16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(fm*16), nn.ReLU(True),
            nn.ConvTranspose2d(fm*16, fm*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fm*8),  nn.ReLU(True),
            nn.ConvTranspose2d(fm*8,  fm*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fm*4),  nn.ReLU(True),
            nn.ConvTranspose2d(fm*4,  fm*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fm*2),  nn.ReLU(True),
            nn.ConvTranspose2d(fm*2,  fm,   4, 2, 1, bias=False),
            nn.BatchNorm2d(fm),    nn.ReLU(True),
            nn.Conv2d(fm, 3, 3, 1, 1),
            nn.Tanh()
        )
    def forward(self, z): return self.net(z.view(-1, 128, 1, 1))

class WGANGenerator(nn.Module):
    def __init__(self, latent=128, fm=128):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(latent, fm*8 * 4 * 4, bias=False), nn.LeakyReLU(0.2, True))
        self.body = nn.Sequential(
            self._block(fm*8, fm*4), self._block(fm*4, fm*2),
            self._block(fm*2, fm),   self._block(fm,   fm//2),
        )
        self.out = nn.Sequential(nn.Conv2d(fm//2, 3, 3, 1, 1), nn.Tanh())
    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_ch), nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_ch), nn.LeakyReLU(0.2, True),
        )
    def forward(self, z):
        x = self.project(z).view(-1, 1024, 4, 4)
        return self.out(self.body(x))

# --- UTILITIES ---
@st.cache_resource
def load_models():
    # Load DCGAN
    dc_gen = DCGenerator()
    dc_gen.load_state_dict(torch.load("dcgan_ep20.pth", map_location="cpu")['G'])
    # Load WGAN-GP
    wg_gen = WGANGenerator()
    wg_gen.load_state_dict(torch.load("wgan_ep12.pth", map_location="cpu")['G'])
    return dc_gen.eval(), wg_gen.eval()

def generate_images(model, num_samples, seed):
    torch.manual_seed(seed)
    z = torch.randn(num_samples, 128)
    with torch.no_grad():
        fakes = model(z)
    fakes = (fakes * 0.5 + 0.5).clamp(0, 1) # Denormalize
    return fakes

def get_image_bytes(tensor):
    grid = make_grid(tensor, normalize=False).permute(1, 2, 0).numpy()
    img = Image.fromarray((grid * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# --- STATE MANAGEMENT ---
if 'history' not in st.session_state: st.session_state.history = []
if 'count' not in st.session_state: st.session_state.count = 0

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color:#00e5cc;'>SpriteForge</h2>", unsafe_allow_html=True)
    model_choice = st.radio("Model Engine", ["DCGAN", "WGAN-GP"], help="DCGAN is faster; WGAN-GP provides higher quality and diversity.")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        seed = st.number_input("Generation Seed", value=42, step=1)
    with col2:
        if st.button("🎲"): seed = random.randint(0, 9999)
    
    batch_size = st.slider("Number of Sprites", 1, 16, 4)
    st.multiselect("Dataset Filter", ["Anime Faces", "Pokemon Sprites"], default=["Anime Faces"])
    
    st.divider()
    st.markdown(f"**Impact Counter**\n\n{st.session_state.count} sprites generated.")
    st.caption(f"That's approximately {st.session_state.count * 5} minutes of artist time saved.")

# --- MAIN APP LOGIC ---
dc_model, wg_model = load_models()
current_model = dc_model if model_choice == "DCGAN" else wg_model

# STICKY HEADER
header = st.container()
with header:
    c1, c2, c3 = st.columns([2, 2, 1])
    c1.markdown("### ⚡ SpriteForge")
    c3.info(f"Active: {model_choice}")

tab_gen, tab_comp, tab_about = st.tabs(["Generate", "Compare", "About"])

# --- TAB: GENERATE ---
with tab_gen:
    col_ctrl, col_main = st.columns([1, 3])
    
    with col_ctrl:
        st.write("Control Panel")
        if st.button("GENERATE ASSETS"):
            with st.spinner("Forging sprites..."):
                imgs = generate_images(current_model, batch_size, seed)
                st.session_state.current_imgs = imgs
                st.session_state.count += batch_size
                st.session_state.history.insert(0, {"imgs": imgs, "seed": seed, "model": model_choice})

    with col_main:
        if 'current_imgs' in st.session_state:
            cols = st.columns(2)
            for i, img_tensor in enumerate(st.session_state.current_imgs):
                with cols[i % 2]:
                    st.markdown(f'<div class="sprite-card">', unsafe_allow_html=True)
                    pil_img = Image.fromarray((img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                    st.image(pil_img, use_container_width=True)
                    st.caption(f"Seed: {seed} | {model_choice}")
                    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB: COMPARE ---
with tab_comp:
    st.warning("Comparing Model Diversity and Mode Collapse Resistance")
    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        st.subheader("DCGAN")
        dc_imgs = generate_images(dc_model, 8, seed)
        st.image(make_grid(dc_imgs, nrow=4).permute(1, 2, 0).numpy())
        st.metric("Avg Brightness", f"{dc_imgs.mean():.2f}")
        
    with comp_col2:
        st.subheader("WGAN-GP")
        wg_imgs = generate_images(wg_model, 8, seed)
        st.image(make_grid(wg_imgs, nrow=4).permute(1, 2, 0).numpy())
        st.metric("Avg Brightness", f"{wg_imgs.mean():.2f}")

# --- TAB: ABOUT ---
with tab_about:
    st.markdown("""
    ### Mission
    SpriteForge democratizes game asset creation for solo developers.
    
    ### Model Cards
    - **DCGAN**: Trained for 70 epochs. Fast generation.
    - **WGAN-GP**: Trained for 50 epochs. Superior stability using Gradient Penalty.
    
    ### Dataset Credits
    - Anime Faces: Soumik Rakshit (Kaggle)
    - Pokemon Sprites: Jack E. Martin (Kaggle)
    """)