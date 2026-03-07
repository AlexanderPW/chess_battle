from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import os

mongo_uri = os.getenv("MONGO_URI")
print(f"Testing with URI: {mongo_uri}")

if not mongo_uri:
    print("ERROR: MONGO_URI not set in .env")
    exit(1)

try:
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ismaster")
    print("✓ Connected to MongoDB")
    
    db = client.chess_battle
    collection = db.games
    print(f"✓ Database 'chess_battle' accessible")
    print(f"✓ Collection 'games' accessible")
    print(f"✓ Current count: {collection.count_documents({})}")
    
except ConnectionFailure as e:
    print(f"✗ Connection failed: {e}")
except ServerSelectionTimeoutError as e:
    print(f"✗ Server timeout: {e}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
