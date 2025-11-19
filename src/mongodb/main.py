from pymongo import MongoClient
from dotenv import load_dotenv
import os
load_dotenv()
mongo_url=os.getenv("MONGO_URL")
mongo_user=os.getenv("MONGO_USER")
mongo_pass=os.getenv("MONGO_PASS")
mongo_auth=os.getenv("AUTH_SOURCE")
mongo_port=os.getenv("MONGO_PORT")


print(mongo_url)
uri = f"""mongodb://{mongo_url}:{mongo_port}"""
client=MongoClient(uri)
try:
    # List all databases
    dbs = client.list_database_names()
    print("Connected successfully!")
    print("Databases:", dbs)

    # Create test database and collection
    test_db = client["test_db"]
    test_col = test_db["test_collection"]

    # Insert a document
    result = test_col.insert_one({"name": "Hooman", "role": "tester"})
    print("Inserted document ID:", result.inserted_id)

    # Read it back
    doc = test_col.find_one({"name": "Hooman"})
    print("Read document:", doc)

except Exception as e:
    print("Connection failed:", e)