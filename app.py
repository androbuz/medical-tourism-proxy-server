import os
import tempfile

from flask import Flask, request, jsonify, redirect
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

# health route
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": True
    })

# prediction route pointing to hugging face
@app.route("/predict", methods=["POST"])
def predict():
    # The medical tourism server expects an image and a question_type
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

        # Call the HF space predict endpoint with both image and question_type
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

# download apk route directing to B2
@app.route("/download", methods=["GET"])
def download():
    url = s3.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": B2_BUCKET,
            "Key": B2_APK_KEY,
        },
        ExpiresIn=300,
    )
    return redirect(url)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(
        host="0.0.0.0",
        port=port
    )
