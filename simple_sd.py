"""
Simple Stable Diffusion Interface
"""
import os
import sys

def main():
    # Import local generator
    try:
        from local_sd_generator import LocalStableDiffusion
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Install: pip install -r requirements.txt")
        return
    
    print("🎨 AI Art Generator - Simple Mode")
    print("=================================")
    
    # Initialize
    generator = LocalStableDiffusion()
    
    # Load model
    print("🔄 Loading AI model...")
    if not generator.load_model():
        print("❌ Could not load model")
        return
    
    print("✅ Ready! Type 'quit' to exit")
    print("💡 Example: 'a beautiful sunset over mountains'")
    print()
    
    # Main loop
    while True:
        try:
            prompt = input("🎨 Describe your art: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Happy creating!")
                break
            
            if not prompt:
                continue
            
            print("⏳ Generating...")
            image, filename = generator.generate_image(prompt)
            
            if image:
                print(f"✅ Done! Check: {filename}")
            else:
                print("❌ Failed to generate")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()