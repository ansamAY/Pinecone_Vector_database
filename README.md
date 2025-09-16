# Image & Text Semantic Search with Vector Database

##  Overview

This project demonstrates how to use **embeddings + vector databases**
to enable **semantic search** for both text and images.\
Instead of relying on keyword-based search, embeddings allow searching
**by meaning**.

We use:\
- **CLIP (OpenAI)** → to generate embeddings for text & images.\
- **Pinecone** → as the vector database to store and query embeddings.\
- **SentenceTransformers** (optional) → to generate text embeddings.

------------------------------------------------------------------------

##  Workflow

1.  **Input Data**: images, text, or video.\
2.  **Embedding Model**: convert them into vector representations
    (arrays of numbers).
    -   Example:
        -   `"apple"` → `[2.55, 6.77, 1.55]`\
        -   `"car"` → `[1.25, 3.98, 5.44]`\
3.  **Vector Database (Pinecone)**:
    -   Store embeddings with metadata (e.g., filename, description).\
    -   Perform fast similarity search.\
4.  **Search**:
    -   Query with text (e.g., `"cute puppy"`) or image (e.g.,
        `new_dog.jpg`).\
    -   Query is embedded → compared with stored embeddings → closest
        matches are returned.

------------------------------------------------------------------------

##  Tech Stack

-   **Python 3.11+**
-   [Transformers](https://huggingface.co/transformers/) (`CLIPModel`,
    `CLIPProcessor`)
-   [SentenceTransformers](https://www.sbert.net/)
-   [Pinecone](https://www.pinecone.io/)
-   [Torch](https://pytorch.org/)
-   [Pillow](https://pillow.readthedocs.io/)

------------------------------------------------------------------------

##  Project Structure

    .
    ├── vector_search_text.py   # Semantic search using text embeddings
    ├── image_search.py         # Semantic search with text or image
    ├── image_search_only.py    # Search by image only
    ├── requirements.txt        # Dependencies
    └── README.md               # Project documentation

------------------------------------------------------------------------

##  Examples

###  Text Search

``` python
query = "comfortable sneakers for sport"
# Returns → "Red running shoes for men"
```

###  Image Search

``` python
query_image = "new_dog.jpg"
# Returns → closest match: "dog.jpg"
```

------------------------------------------------------------------------

##  Embedding Models Examples

-   **Text**
    -   OpenAI `text-embedding-ada-002`\
    -   SentenceTransformers `all-MiniLM-L6-v2`\
-   **Image**
    -   OpenAI CLIP\
    -   ViT (Vision Transformer)\
-   **Multimodal**
    -   CLIP (text + image in same space)

------------------------------------------------------------------------

##  Benefits

-   Semantic (meaning-based) search.\
-   Cross-modal search (text ↔ image).\
-   Scalable with Pinecone.\
-   Ideal for e-commerce, recommendations, moderation, and search
    engines.
