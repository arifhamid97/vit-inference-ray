# Inference Using Ray Serve

This guide provides a step-by-step process to set up and use Ray Serve for serving the fine-tuned Vision Transformer (ViT) model for medicinal plant classification.

### Step 1: Install ray serve
```bash
pip install ray[serve]
```

### Step 2: Running the Serve Script
To start the Ray Serve server, run the `serve.py` script:

```bash
python serve.py
```
This will setup local ray cluster on your laptop
and start a ray proxy on `http://localhost:5000`.

Alternatively you can also deploy the serve config to existing ray cluster
```bash
serve run serve_config.yaml
```

## Step 3: Making Inference Requests
run the ``inference.ipynb`` notebook




## Summary
- Set up Ray Serve to deploy the trained Vision Transformer model.
- Use FastAPI to handle HTTP requests for inference.
- Make predictions by sending image data to the Ray Serve endpoint.
