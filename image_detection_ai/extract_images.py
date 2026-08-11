import pymongo
import os

# Connect to your MongoDB Image Database
client = pymongo.MongoClient("mongodb://192.168.1.33:27017/")
db = client["marg_sahayak_images"]
collection = db["images"]

# Create a folder to save the images
output_dir = "real_db_images"
os.makedirs(output_dir, exist_ok=True)

print("Fetching the latest 20 images from your database...")
# Get the most recent 20 images
images = collection.find().sort("createdAt", -1).limit(20)

count = 0
for img in images:
    try:
        # Get the original image name (or generate one)
        name = img.get("name", f"image_{count}.jpg")
        
        # Extract the binary data
        img_dict = img.get("img", {})
        data = img_dict.get("data")
        
        if data:
            filepath = os.path.join(output_dir, name)
            with open(filepath, "wb") as f:
                f.write(data)
            print(f"✅ Saved: {name}")
            count += 1
    except Exception as e:
        print(f"❌ Error saving image: {e}")

print(f"\nDone! Successfully extracted {count} real images into the '{output_dir}' folder.")
