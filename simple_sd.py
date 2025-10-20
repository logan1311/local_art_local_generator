# src/simple_sd.py
import os
import sys

def main():
    print("🎨 AI Art Generator - High Resolution")
    print("====================================")
    
    try:
        from local_sd_generator import LocalStableDiffusion
        generator = LocalStableDiffusion()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Install: pip install -r requirements.txt")
        return
    
    # Load model
    print("🔄 Loading AI model...")
    if not generator.load_model(low_memory=True):
        print("❌ Model loading failed")
        return
    
    print("✅ Ready! Type 'quit' to exit")
    print("\n💡 Size Options:")
    print("   small  = 512x512  (fastest)")
    print("   medium = 768x768  (recommended)")
    print("   large  = 1024x1024 (best quality)")
    print("   portrait = 768x1024")
    print("   landscape = 1024x768")
    print("\n🌟 Example: 'medium a beautiful sunset'")
    print()
    
    # Main loop
    while True:
        try:
            user_input = input("🎨 Enter size and description: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Happy creating!")
                break
            
            if not user_input:
                continue
            
            # Parse size and prompt
            parts = user_input.split(' ', 1)
            if len(parts) == 1:
                # No size specified, use medium
                size = "medium"
                prompt = parts[0]
            else:
                size = parts[0].lower()
                prompt = parts[1]
            
            # Validate size
            valid_sizes = ['small', 'medium', 'large', 'portrait', 'landscape']
            if size not in valid_sizes:
                print(f"❌ Invalid size. Using 'medium' instead.")
                size = "medium"
            
            print(f"⏳ Generating {size} image...")
            
            if size in ['small', 'medium', 'large']:
                image, filename = generator.generate_high_quality(prompt, size)
            else:
                # Custom sizes
                if size == "portrait":
                    image, filename = generator.generate_image(prompt, width=768, height=1024)
                else:  # landscape
                    image, filename = generator.generate_image(prompt, width=1024, height=768)
            
            if image:
                print(f"✅ Success! Check: {filename}")
                print(f"📊 Image size: {image.size}")
                print(f"📁 Location: {os.path.abspath(filename)}")
            else:
                print("❌ Generation failed - try a smaller size or different prompt")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
