"""Modal GPU entrypoint for the retail shelf prediction API."""

import modal


app = modal.App("tnsqai-retail-shelf")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install_from_requirements("requirements-inference.txt")
    .env({"HF_HOME": "/opt/huggingface"})
    .run_commands(
        "python -c \"import timm; "
        "model = timm.create_model('vit_small_patch14_dinov2.lvd142m', "
        "pretrained=True, num_classes=0); "
        "assert model.num_features == 384\""
    )
    .env({"HF_HUB_OFFLINE": "1", "YOLO_WEIGHTS_PATH": "/models/best.pt"})
    .add_local_python_source("api", "inference", copy=True)
    .add_local_file("weights/best.pt", "/models/best.pt", copy=True)
)


@app.function(image=image, gpu="T4", timeout=600, max_containers=1)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def serve():
    """Load both models once per container and serve the existing FastAPI app."""

    from api.main import create_app
    from inference.config import InferenceConfig
    from inference.model_loader import load_models
    from inference.pipeline import ShelfInferencePipeline

    config = InferenceConfig.from_env()
    models = load_models(config)
    return create_app(ShelfInferencePipeline(models, config))
