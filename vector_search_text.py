from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import time

INDEX_NAME = "product-search"
REGION = "us-east-1"
DIM = 384

# 1) Embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2) Pinecone client
pc = Pinecone(api_key="pcsk_5AfaQ4_F7Yg8xTf7zMdhNApTrourR2HmumPEszHnf4f6k7EiCsNVuARA2YPKMaw4b1u8K4")

# 3) Create index if not exists
if INDEX_NAME not in [x["name"] for x in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=REGION)
    )

# Wait until ready
while not pc.describe_index(INDEX_NAME)["status"]["ready"]:
    time.sleep(1)
print("Index is ready!")

index = pc.Index(INDEX_NAME)

# 4) Example products
products = [
    "Red running shoes for men",
    "Elegant black dress for women",
    "Wireless noise-cancelling headphones",
    "Smartphone with high-resolution camera",
    "Wooden dining table with 6 chairs",
    "Laptop with long battery life",
]

# 5) Correct upsert format
vectors = [
    {
        "id": f"prod-{i}",
        "values": model.encode(p).tolist(),
        "metadata": {"description": p}
    }
    for i, p in enumerate(products)
]

up_res = index.upsert(vectors=vectors)
print("Upsert response:", up_res)

# 6) Verify stats
stats = index.describe_index_stats()
print("Index stats:", stats)

# 7) Query
query = "comfortable sneakers for sport"
qvec = model.encode(query).tolist()
results = index.query(vector=qvec, top_k=1, include_metadata=True)

print("\n�� Search Results for:", query)
for m in results["matches"]:
    print(f"Score: {m['score']:.3f} | {m['metadata']['description']}")

