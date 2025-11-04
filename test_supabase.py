import os
from dotenv import load_dotenv
load_dotenv()

from supabase import create_client

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_KEY")
bucket = os.getenv("SUPABASE_BUCKET", "media")

if not url or not key:
    print("ERROR: SUPABASE_URL or SUPABASE_SERVICE_KEY not set in .env")
    exit(1)

print(f"Connecting to Supabase: {url}")
print(f"Bucket: {bucket}")

try:
    sb = create_client(url, key)
    
    # Test upload
    p = "static/uploads"
    os.makedirs(p, exist_ok=True)
    fn = os.path.join(p, "test.txt")
    with open(fn, "wb") as f:
        f.write(b"hello world")
    
    print(f"\nUploading test file to {bucket}/test/test.txt...")
    with open(fn, "rb") as f:
        sb.storage.from_(bucket).upload(
            file=f,
            path="test/test.txt",
            file_options={"contentType": "text/plain"}
        )
    
    # Get public URL
    result = sb.storage.from_(bucket).get_public_url("test/test.txt")
    print(f"\nSUCCESS! Public URL: {result}")
    print("\nCheck your Supabase bucket - you should see test/test.txt")
    print("If this URL opens in your browser, uploads are working!")
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()

