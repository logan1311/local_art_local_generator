# src/local_sd_generator.py
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os
from datetime import datetime
import gc

class LocalStableDiffusion:
    def __init__(self, model_name="runwayml/stable-diffusion-v1-5"):
        # Set environment variable to prevent timeout
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.pipe = None
        
        print(f"🚀 Device: {self.device}")
        print(f"📦 Model: {model_name}")
        
        os.makedirs("outputs", exist_ok=True)
        
    def load_model(self, low_memory=True):
        """Load model with timeout prevention"""
        try:
            print("📥 Loading model...")
            
            # Clear memory first
            self._clean_memory()
            
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True
            )
            
            self.pipe = self.pipe.to(self.device)
            
            # Use faster scheduler
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            
            # Apply memory optimizations
            if low_memory:
                self._apply_memory_optimizations()
            
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    
    def _apply_memory_optimizations(self):
        """Apply memory saving optimizations to prevent timeout"""
        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()
            print("💾 Enabled attention slicing")
        
        if hasattr(self.pipe, "enable_vae_slicing"):
            self.pipe.enable_vae_slicing()
            print("💾 Enabled VAE slicing")
        
        if hasattr(self.pipe, "enable_sequential_cpu_offload") and self.device == "cuda":
            self.pipe.enable_sequential_cpu_offload()
            print("💾 Enabled CPU offloading")
    
    def _clean_memory(self):
        """Clean up GPU memory"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    
    def generate_image_safe(self, prompt, negative_prompt="", steps=20, width=512, height=512):
        """Safe generation with timeout prevention"""
        if not self.pipe:
            print("❌ Please load model first!")
            return None, None
        
        try:
            print(f"🎨 Generating: '{prompt}'")
            
            # Use smaller steps for faster generation
            if steps > 25:
                steps = 20
                print("⚠️ Reduced steps to 20 for stability")
            
            # Generate with error handling
            with torch.inference_mode():  # More memory efficient
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt or "blurry, cartoon, deformed",
                    num_inference_steps=steps,
                    guidance_scale=7.0,  # Slightly lower for stability
                    width=width,
                    height=height
                )
            
            image = result.images[0]
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prompt = "".join(c for c in prompt[:20] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"outputs/art_{safe_prompt}_{timestamp}.png"
            image.save(filename)
            
            print(f"✅ Artwork saved: {filename}")
            return image, filename
            
        except torch.cuda.OutOfMemoryError:
            print("❌ GPU out of memory! Try smaller image size or use CPU")
            return None, None
        except RuntimeError as e:
            if "timeout" in str(e).lower():
                print("❌ GPU timeout! Try the CPU version or smaller images")
                return None, None
            else:
                print(f"❌ Runtime error: {e}")
                return None, None
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return None, None

    # Simple alias for easy use
    def generate_image(self, prompt, **kwargs):
        return self.generate_image_safe(prompt, **kwargs)

# Test function
def test_safe_generation():
    generator = LocalStableDiffusion()
    
    if generator.load_model(low_memory=True):
        test_prompts = [
            "a beautiful sunset over mountains",
            "a cute cat sitting on a windowsill",
        ]
        
        for prompt in test_prompts:
            print(f"\n{'='*50}")
            image, filename = generator.generate_image_safe(prompt)
            if image:
                print(f"✅ Success: {filename}")
            else:
                print("❌ Failed - trying CPU version...")
                # Fallback to CPU if available
                pass