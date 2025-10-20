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
    
    def _clean_memory(self):
        """Clean up GPU memory"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    
    def generate_image(self, prompt, negative_prompt="", steps=25, width=768, height=768, upscale=False):
        """Generate higher resolution images"""
        if not self.pipe:
            print("❌ Please load model first!")
            return None, None
        
        try:
            print(f"🎨 Generating: '{prompt}'")
            print(f"📐 Size: {width}x{height}")
            
            # Safety check for memory
            if width > 1024 or height > 1024:
                print("⚠️ Reducing size to 1024x1024 for memory safety")
                width = min(width, 1024)
                height = min(height, 1024)
            
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or "blurry, cartoon, deformed, low quality",
                num_inference_steps=steps,
                guidance_scale=7.5,
                width=width,
                height=height
            )
            
            image = result.images[0]
            
            # Upscale if requested
            if upscale:
                image = self._upscale_image(image)
                print("🔍 Image upscaled 2x")
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            size_suffix = f"{width}x{height}"
            safe_prompt = "".join(c for c in prompt[:20] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"outputs/art_{safe_prompt}_{size_suffix}_{timestamp}.png"
            image.save(filename, quality=95)
            
            print(f"✅ High-res artwork saved: {filename}")
            print(f"📊 Final size: {image.size}")
            return image, filename
            
        except torch.cuda.OutOfMemoryError:
            print("❌ GPU out of memory! Try smaller image size")
            return self._fallback_generate(prompt)
        except RuntimeError as e:
            if "timeout" in str(e).lower() or "memory" in str(e).lower():
                print("❌ GPU issue! Trying smaller size...")
                return self._fallback_generate(prompt)
            else:
                print(f"❌ Runtime error: {e}")
                return None, None
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return None, None
    
    def _fallback_generate(self, prompt):
        """Fallback to smaller size if memory fails"""
        print("🔄 Trying smaller size (512x512)...")
        try:
            result = self.pipe(
                prompt=prompt,
                negative_prompt="blurry, cartoon, deformed, low quality",
                num_inference_steps=20,
                guidance_scale=7.5,
                width=512,
                height=512
            )
            
            image = result.images[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"outputs/art_fallback_{timestamp}.png"
            image.save(filename, quality=95)
            
            print(f"✅ Fallback artwork saved: {filename}")
            return image, filename
        except Exception as e:
            print(f"❌ Fallback also failed: {e}")
            return None, None
    
    def _upscale_image(self, image, scale_factor=2):
        """Simple upscaling for better quality"""
        new_size = (image.width * scale_factor, image.height * scale_factor)
        return image.resize(new_size, Image.Resampling.LANCZOS)
    
    def generate_high_quality(self, prompt, size="large"):
        """Convenience method for high-quality generation"""
        sizes = {
            "small": (512, 512),
            "medium": (768, 768),
            "large": (1024, 1024),
            "portrait": (768, 1024),
            "landscape": (1024, 768)
        }
        
        width, height = sizes.get(size, (768, 768))
        
        print(f"🎨 Generating {size} image: {width}x{height}")
        return self.generate_image(prompt, width=width, height=height)
    
    def generate_portrait(self, description, high_quality=True):
        """Generate realistic portrait"""
        prompt = f"professional portrait photography, {description}, detailed eyes, natural skin texture, studio lighting, 85mm lens, high detail"
        size = (1024, 1024) if high_quality else (768, 768)
        return self.generate_image(prompt, "blurry, cartoon, anime, plastic, deformed", width=size[0], height=size[1])
    
    def generate_landscape(self, description, high_quality=True):
        """Generate realistic landscape"""
        prompt = f"landscape photography, {description}, golden hour, natural lighting, ultra detailed, National Geographic style, 8k"
        size = (1024, 768) if high_quality else (768, 512)
        return self.generate_image(prompt, "cartoon, painting, drawing, blurry, oversaturated", width=size[0], height=size[1])

# Test with higher resolution
def test_high_res():
    generator = LocalStableDiffusion()
    
    if generator.load_model(low_memory=True):
        test_prompts = [
            "a beautiful sunset over mountains, landscape photography, highly detailed",
            "a majestic wolf in snowy forest, wildlife photography, sharp focus",
        ]
        
        for prompt in test_prompts:
            print(f"\n{'='*50}")
            # Try high quality first, fallback if needed
            image, filename = generator.generate_high_quality(prompt, "medium")
            if image:
                print(f"✅ Success: {filename}")
            else:
                print("❌ High quality failed, trying standard...")
                image, filename = generator.generate_image(prompt, width=512, height=512)
