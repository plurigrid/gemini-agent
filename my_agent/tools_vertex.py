"""Real Vertex AI tools using google-cloud-aiplatform."""

import json
import os

from google.cloud import aiplatform


PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "plurigrid")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

_initialized = False


def _ensure_init():
    global _initialized
    if not _initialized:
        aiplatform.init(project=PROJECT, location=LOCATION)
        _initialized = True


def vertex_predict(model_endpoint: str, instances: str) -> dict:
    """Send prediction request to a Vertex AI endpoint.

    Args:
        model_endpoint: Vertex AI endpoint ID or full resource name.
        instances: JSON string of prediction instances.
    """
    _ensure_init()
    try:
        endpoint = aiplatform.Endpoint(model_endpoint)
        parsed = json.loads(instances)
        if not isinstance(parsed, list):
            parsed = [parsed]
        response = endpoint.predict(instances=parsed)
        return {
            "predictions": [str(p) for p in response.predictions],
            "deployed_model_id": response.deployed_model_id,
            "status": "value",
        }
    except Exception as e:
        return {"error": str(e), "status": "contradiction"}


def vertex_pipeline_status(pipeline_job_name: str) -> dict:
    """Check status of a Vertex AI Pipeline job.

    Args:
        pipeline_job_name: Full resource name or display name of the pipeline job.
    """
    _ensure_init()
    try:
        job = aiplatform.PipelineJob.get(pipeline_job_name)
        return {
            "job": job.display_name,
            "state": job.state.name,
            "create_time": str(job.create_time),
            "trit": "zero",
            "status": "value",
        }
    except Exception as e:
        return {"error": str(e), "status": "contradiction"}


def vertex_list_endpoints() -> dict:
    """List all Vertex AI endpoints in the project."""
    _ensure_init()
    try:
        endpoints = aiplatform.Endpoint.list()
        return {
            "endpoints": [
                {"name": ep.display_name, "resource": ep.resource_name}
                for ep in endpoints[:20]
            ],
            "count": len(endpoints),
            "status": "value",
        }
    except Exception as e:
        return {"error": str(e), "status": "contradiction"}


def vertex_list_models() -> dict:
    """List all models in the Vertex AI Model Registry."""
    _ensure_init()
    try:
        models = aiplatform.Model.list()
        return {
            "models": [
                {"name": m.display_name, "resource": m.resource_name}
                for m in models[:20]
            ],
            "count": len(models),
            "status": "value",
        }
    except Exception as e:
        return {"error": str(e), "status": "contradiction"}


def gemini_generate(prompt: str, model: str = "gemini-2.0-flash") -> dict:
    """Generate text using Gemini via google.genai.

    Args:
        prompt: The prompt to send to Gemini.
        model: Model ID to use.
    """
    try:
        from google import genai
        client = genai.Client(
            vertexai=True,
            project=PROJECT,
            location=LOCATION,
        )
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        return {
            "model": model,
            "response": response.text,
            "status": "value",
        }
    except Exception as e:
        return {"error": str(e), "status": "contradiction"}
