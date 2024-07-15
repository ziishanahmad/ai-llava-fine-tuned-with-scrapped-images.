# AI LLaVA Fine-Tuned with Scrapped Images

## Overview
This repository contains code for an AI model that scrapes images and their alt tags from a website to create a dataset. The dataset is then used to fine-tune a LLaVA VisionEncoderDecoderModel for image captioning. The project leverages Hugging Face's transformers library to generate descriptive captions for the images, showcasing the integration of web scraping and advanced AI techniques for practical applications.

## Features
- **Web Scraping:** Automatically downloads images and their alt tags from a specified website.
- **Dataset Creation:** Formats the scrapped data into a JSON dataset suitable for training.
- **Model Fine-Tuning:** Fine-tunes a pre-trained VisionEncoderDecoderModel for image captioning.
- **Hugging Face Integration:** Uses Hugging Face's transformers library for model training and inference.

## Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ziishanahmad/ai-llava-fine-tuned-with-scrapped-images.git
   cd ai-llava-fine-tuned-with-scrapped-images
   ```

2. **Install Dependencies:**
   Ensure you have Python 3.7+ and install the required packages:
   ```bash
   pip install transformers datasets scikit-learn pillow beautifulsoup4 accelerate
   ```

## Usage
### Download Images and Create Dataset
The following script scrapes images from the specified website and creates a JSON dataset.
```python
import os
import requests
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque

def download_images_and_create_json(url, output_dir, json_path, max_depth=2, max_images=10):
    os.makedirs(output_dir, exist_ok=True)
    data = []
    visited_urls = set()
    queue = deque([(url, 0)])
    image_count = 0

    while queue and image_count < max_images:
        current_url, depth = queue.popleft()
        if depth > max_depth or current_url in visited_urls:
            continue
        visited_urls.add(current_url)

        try:
            response = requests.get(current_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            image_tags = soup.find_all('img')

            for img in image_tags:
                img_url = img.get('src')
                alt_tag = img.get('alt', '')

                if img_url:
                    img_url = urljoin(current_url, img_url)
                    img_name = img_url.split('/')[-1]
                    img_ext = img_name.split('.')[-1]

                    if img_ext not in ['jpg', 'png', 'jpeg']:
                        continue

                    img_response = requests.get(img_url, stream=True)
                    img_response.raw.decode_content = True

                    img_path = os.path.join(output_dir, img_name)
                    with open(img_path, 'wb') as f:
                        f.write(img_response.content)

                    data.append({
                        "id": f"image_{image_count}",
                        "image": img_path,
                        "conversations": [
                            {"from": "human", "value": "What is shown in the image?"},
                            {"from": "gpt", "value": alt_tag}
                        ]
                    })
                    image_count += 1
                    if image_count >= max_images:
                        break

                    print(f"Downloaded: {img_path}, Alt Tag: {alt_tag}")

        except Exception as e:
            print(f"Error fetching {current_url}: {e}")

        if depth < max_depth:
            for link in soup.find_all('a', href=True):
                next_url = urljoin(current_url, link['href'])
                if urlparse(next_url).netloc == urlparse(url).netloc:
                    queue.append((next_url, depth + 1))

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

# Example usage
website_url = 'https://www.apple.com/iphone/'
output_dir = '/content/applescrapper2'
json_path = '/content/dataset.json'
download_images_and_create_json(website_url, output_dir, json_path)
```

### Fine-Tune the Model
The following script fine-tunes the VisionEncoderDecoderModel using the dataset created in the previous step.
```python
import torch
from PIL import Image
from datasets import Dataset
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the dataset from JSON
def load_json_dataset(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    images = [item["image"] for item in data]
    captions = [item["conversations"][1]["value"] for item in data]
    return Dataset.from_dict({'image_path': images, 'caption': captions})

train_dataset = load_json_dataset(json_path)
val_dataset = load_json_dataset(json_path)

# Load the pre-trained model, feature extractor, and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Preprocess function
def preprocess_function(examples):
    images = [Image.open(image).convert("RGB") for image in examples["image_path"]]
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    captions = tokenizer(examples["caption"], padding="max_length", truncation=True, return_tensors="pt")
    return {"pixel_values": pixel_values, "labels": captions.input_ids}

# Preprocess datasets
train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["image_path", "caption"])
val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=["image_path", "caption"])

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3
)

# Initialize the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=default_data_collator,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("/content/fine_tuned_model")
tokenizer.save_pretrained("/content/fine_tuned_tokenizer")
feature_extractor.save_pretrained("/content/fine_tuned_feature_extractor")
```

### Testing the Model
The following script tests the fine-tuned model using an image from the dataset.
```python
import matplotlib.pyplot as plt

# Load the fine-tuned model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("/content/fine_tuned_model").to(device)
feature_extractor = ViTImageProcessor.from_pretrained("/content/fine_tuned_feature_extractor")
tokenizer = AutoTokenizer.from_pretrained("/content/fine_tuned_tokenizer")

# Function to predict image caption
def predict_image_caption(image_path, model, feature_extractor, tokenizer):
    img = Image.open(image_path).convert("RGB")
    pixel_values = feature_extractor(images=img, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        output_ids = model.generate(pixel_values)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return img, caption

# Predict and display the caption for a new image
new_image_path = '/content/applescrapper2/cinematic_mode__eqivqhqqnj42_large.png'
img, caption = predict_image_caption(new_image_path, model, feature_extractor, tokenizer)

plt.imshow(img)
plt.title(f"Predicted Caption: {caption}")
plt.axis('off')
plt.show()
```

## Results
The fine-tuned model generates descriptive captions for images scrapped from the specified website. The integration of web scraping and advanced AI techniques demonstrates a practical application of fine-tuning pre-trained models with custom datasets.

## Acknowledgements
This project leverages Hugging Face's transformers library and the LLaVA VisionEncoderDecoderModel. Special thanks to the developers and contributors of these tools.

## License
This project is licensed under the MIT License.
