import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pinecone import Pinecone, ServerlessSpec
import time

# 1) Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2) Connect to Pinecone
pc = Pinecone(api_key="pcsk_5AfaQ4_F7Yg8xTf7zMdhNApTrourR2HmumPEszHnf4f6k7EiCsNVuARA2YPKMaw4b1u8K4")
INDEX_NAME = "image-search"

# 3) Create index if not exists
if INDEX_NAME not in [x["name"] for x in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=512,  # CLIP image embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # free plan region
    )

# Wait until index is ready
while not pc.describe_index(INDEX_NAME)["status"]["ready"]:
    time.sleep(1)
print("✅ Index is ready!")

index = pc.Index(INDEX_NAME)

# 4) Insert example images (do this once)
image_files = ["cat.jpg", "dog.jpg", "car.jpg"]  # replace with your own files

vectors = []
for i, file in enumerate(image_files):
    img = Image.open(file)
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        emb = model.get_image_features(**inputs).squeeze().tolist()
    vectors.append({"id": f"img-{i}", "values": emb, "metadata": {"filename": file}})

if vectors:
    up_res = index.upsert(vectors=vectors)
    print("Upsert response:", up_res)

# 5) Search by a new image
query_image = "new_car.jpg"  # replace with any image file path
img = Image.open(query_image)
inputs = processor(images=img, return_tensors="pt")
with torch.no_grad():
    qvec = model.get_image_features(**inputs).squeeze().tolist()

results = index.query(vector=qvec, top_k=1, include_metadata=True)

print("\n🖼️ Search Results for image:", query_image)
for m in results["matches"]:
    print(f"Score: {m['score']:.3f} | {m['metadata']['filename']}")



