import openai
import os

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import os
from dotenv import load_dotenv
load_dotenv()


openai.api_type = "azure"
openai.api_base = "https://096290-oai.openai.azure.com"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_version = "2023-05-15"

def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        deployment_id="team10-embedding"
    )
    return response['data'][0]['embedding']





# Upload inventory item
def index_inventory_item(item):
    vector = get_embedding(item["name"])
    qdrant.upsert(
        collection_name="inventory_vectors",
        points=[
            PointStruct(
                id=item["index"],
                vector=vector,
                payload=item
            )
        ]
    )

def delete_inventory_item(item_id):
    qdrant.delete(
        collection_name="inventory_vectors",
        points_selector={"points": [item_id]}
    )
def embed_inventory_data(inventory):
    for item in inventory:
        index_inventory_item(item)
if __name__ == "__main__":
    qdrant = QdrantClient(
        url="https://02c351f8-4e7e-40c8-9b4b-c0a4b0bb26cf.us-east-1-0.aws.cloud.qdrant.io:6333",
        api_key=os.getenv("QDRANT_API_KEY")
    )

    print(qdrant.get_collections())

    collection_name = "inventory_vectors"

    if not qdrant.collection_exists(collection_name):
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1536,
                distance=Distance.COSINE
            )
        )

    inventory = pd.read_csv("home_inventory.csv")
    inventory = inventory.to_dict(orient='records')
    embed_inventory_data(inventory)
# from qdrant_client.models import PointStruct

# qdrant.upsert(
#     collection_name="inventory_vectors",
#     points=[
#         PointStruct(
#             id=item["id"],
#             vector=get_embedding(item["name"]),
#             payload=item
#         )
#     ]
# )
# qdrant.delete(
#     collection_name="inventory_vectors",
#     points_selector={"points": [item_id]}
# )
