import os
import tempfile
from flask import Flask, request, jsonify, send_file
from gradio_client import Client, handle_file
import boto3

app = Flask(__name__)

# env variables that will setup a hf space connection
HF_SPACE = os.environ.get("HF_SPACE", "Andro777/med-tourism-vlm")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# B2 credentials loaded as environment variables
B2_KEY_ID = os.environ.get("B2_KEY_ID", "")
B2_APPLICATION_KEY = os.environ.get("B2_APPLICATION_KEY", "")

# B2 configuration
B2_BUCKET = "medical-tourism"
B2_ENDPOINT = "https://s3.eu-central-003.backblazeb2.com"
B2_REGION = "eu-central-003"
B2_APK_KEY = "downloads/app.apk"

# Local cache path for the APK file
LOCAL_APK_PATH = os.path.join(tempfile.gettempdir(), "app_cached.apk")

# S3 client
s3 = boto3.client(
    "s3",
    endpoint_url=B2_ENDPOINT,
    aws_access_key_id=B2_KEY_ID,
    aws_secret_access_key=B2_APPLICATION_KEY,
    region_name=B2_REGION,
)

# HF client
client = Client(
    HF_SPACE,
    token=HF_TOKEN,
    verbose=False
)

def download_apk_from_b2():
    """Helper function to download the latest APK from B2 to the local cache."""
    print("Downloading APK from B2 to cache...")
    s3.download_file(B2_BUCKET, B2_APK_KEY, LOCAL_APK_PATH)
    print("APK cached successfully.")

# health route
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": True
    })

# prediction route pointing to hugging face
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    question_type = request.form.get("question_type", "general")

    if not file:
        return jsonify({
            "success": False,
            "error": "No image file provided"
        }), 400

    temp_path = None

    try:
        suffix = os.path.splitext(
            file.filename or "image.png"
        )[1]

        if not suffix:
            suffix = ".png"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:
            temp_path = temp_file.name
            file.save(temp_path)

        result = client.predict(
            image=handle_file(temp_path),
            question_type=question_type,
            api_name="/predict"
        )

        if isinstance(result, tuple):
            result = result[0]

        if isinstance(result, dict):
            return jsonify(result)

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# download cached apk route
@app.route("/download", methods=["GET"])
def download():
    try:
        # If not cached yet, download it once
        if not os.path.exists(LOCAL_APK_PATH):
            download_apk_from_b2()
            
        return send_file(
            LOCAL_APK_PATH,
            mimetype="application/vnd.android.package-archive",
            as_attachment=True,
            download_name="app.apk"
        )
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to serve APK: {str(e)}"
        }), 500

# endpoint to trigger a fresh download/reload of the APK
@app.route("/reload-apk", methods=["GET", "POST"])
def reload_apk():
    try:
        download_apk_from_b2()
        return jsonify({
            "success": True,
            "message": "APK cache successfully updated from B2."
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to update cache: {str(e)}"
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(
        host="0.0.0.0",
        port=port
    )
