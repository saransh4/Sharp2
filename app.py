from flask import Flask, request, jsonify
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from model_handler import ModelHandler  # <-- Import handler

ModelHandler = ModelHandler()

app = Flask(__name__)

# ----------------------------
# Metrics
# ----------------------------
api_call_counter = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['endpoint', 'method']
)

response_time_histogram = Histogram(
    'api_response_time_seconds',
    'Histogram of API response times',
    ['endpoint', 'method']
)

# ----------------------------
# Dependencies
# ----------------------------
database_connected = True


# ----------------------------
# Health Endpoints
# ----------------------------
@app.route("/health/live", methods=["GET"])
def health_live():
    return jsonify({"status": "alive"}), 200


@app.route("/health/ready", methods=["GET"])
def health_ready():
    if database_connected and ModelHandler.is_loaded():
        return jsonify({"status": "ready"}), 200
    else:
        return jsonify({"status": "not ready"}), 503


@app.route("/health/detailed", methods=["GET"])
def health_detailed():
    details = {
        "app_status": "running",
        "timestamp": time.time(),
        "database_connected": database_connected,
        "model_loaded": ModelHandler.is_loaded(),
    }
    overall_status = 200 if all([database_connected, ModelHandler.is_loaded()]) else 503
    return jsonify(details), overall_status


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "timestamp": time.time(),
    })


# ----------------------------
# Metrics Middleware
# ----------------------------
@app.before_request
def start_timer():
    request.start_time = time.time()


@app.after_request
def track_metrics(response):
    if request.endpoint != 'metrics':
        resp_time = time.time() - request.start_time
        api_call_counter.labels(request.path, request.method).inc()
        response_time_histogram.labels(request.path, request.method).observe(resp_time)
    return response


@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


# ----------------------------
# LLM Inference Endpoint
# ----------------------------
@app.route("/generate", methods=["POST"])
def generate():
    if not ModelHandler.is_loaded():
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    prompts = data.get("prompts", [])

    if not prompts:
        return jsonify({"error": "No prompts provided"}), 400

    try:
        responses = ModelHandler.generate_responses(prompts)
        return jsonify(responses)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------------------
# Run Flask App
# ----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
