import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# Disable OneDNN optimizations in TensorFlow (if using TensorFlow)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load model, feature extractor, and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Set device configuration (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set generation configuration
max_length = 16
num_beams = 4  # Corrected from num_beans to num_beams
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_caption(image_paths):
    images = []
    captions = []
    
    for image_path in image_paths:
        try:
            # Open and process the image
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")
            images.append(i_image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue
    
    # Convert images to pixel values using the feature extractor
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Generate captions for the images
    output_ids = model.generate(pixel_values, **gen_kwargs)

    # Decode the generated token IDs to text (captions)
    captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    captions = [caption.strip() for caption in captions]

    # Output the final generated captions
    for img, caption in zip(images, captions):
        show_image_with_caption(img, caption)

    return captions

def show_image_with_caption(image, caption):
    # Display the image with the caption using Matplotlib
    plt.imshow(image)
    plt.axis('off')  # Hide axes
    plt.title(caption, fontsize=12)
    plt.show()

# Example usage
predict_caption(['nature.jpg'])
