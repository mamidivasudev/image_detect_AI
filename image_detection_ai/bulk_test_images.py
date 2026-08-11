import os
import csv
import torch
from PIL import Image

# Force offline mode to prevent HuggingFace connection timeouts
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import CLIPProcessor, CLIPModel

print("Loading AI model into memory... Please wait.")
MODEL_NAME = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
print("Model loaded successfully!\n")

# Define paths
images_folder = r"D:\Downloads\marg_images"
csv_output_file = r"D:\Downloads\ai_bulk_test_results.csv"

# Generic prompts since we don't know the exact category selected by the citizen in the DB
text_queries = [
    "A photo of damaged outdoor public road infrastructure, such as potholes, cracked bridges, broken walls, or washed out roads.",
    "A photo inside a house, bedroom, kitchen, office, or indoor furniture.",
    "A selfie, person's face, pet, food, paper, receipt, or irrelevant image."
]

print(f"Starting bulk processing of images in: {images_folder}")
print(f"Results will be saved to: {csv_output_file}\n")

image_files = [f for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]
total = len(image_files)

# Prepare CSV file
with open(csv_output_file, mode='w', newline='', encoding='utf-8') as csv_file:
    fieldnames = ['Filename', 'Decision', 'Reason', 'Road_Score_%', 'Indoor_Score_%', 'Random_Score_%']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    count = 0
    for filename in image_files:
        filepath = os.path.join(images_folder, filename)
        count += 1
        print(f"Processing [{count}/{total}]: {filename} ...", end=" ")
        
        try:
            image = Image.open(filepath).convert("RGB")
            
            inputs = processor(text=text_queries, images=image, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=-1).squeeze()
                
            scores = probs.tolist()
            road_score = scores[0]
            indoor_score = scores[1]
            random_score = scores[2]
            
            if road_score > 0.60:
                decision = "APPROVED"
                reason = "Looks like valid road infrastructure."
            elif indoor_score > 0.45 or random_score > 0.45:
                decision = "REJECTED"
                reason = "Image appears to be indoor, fake, or irrelevant."
            else:
                decision = "REVIEW"
                reason = "Image is unclear."
                
            writer.writerow({
                'Filename': filename,
                'Decision': decision,
                'Reason': reason,
                'Road_Score_%': round(road_score * 100, 1),
                'Indoor_Score_%': round(indoor_score * 100, 1),
                'Random_Score_%': round(random_score * 100, 1)
            })
            print(f"{decision} ({round(road_score * 100, 1)}%)")
            
        except Exception as e:
            print(f"ERROR")
            writer.writerow({
                'Filename': filename,
                'Decision': "ERROR",
                'Reason': str(e),
                'Road_Score_%': "",
                'Indoor_Score_%': "",
                'Random_Score_%': ""
            })

print("\n==================================")
print("Finished processing all images!")
print(f"Open the results here: {csv_output_file}")
