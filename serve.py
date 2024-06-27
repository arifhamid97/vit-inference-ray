import logging
from fastapi.responses import JSONResponse
from ray import serve
from fastapi import FastAPI, File, Request, UploadFile
import ray.serve
from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from PIL import Image
import io
import ray

logger = logging.getLogger("ray.serve")


# Initialize FastAPI
app = FastAPI()

# Load the Hugging Face model for sentiment analysis
LOCAL_MODEL_PATH = './.model/vit-medicinal-plant-finetune'
HF_REPO = 'funkepal/vit-medicinal-plant-finetune'


@serve.deployment(num_replicas=1, max_ongoing_requests=10)
@serve.ingress(app)
class MedicinalPlantClassification:
    def __init__(self):
        self.processor = ViTImageProcessor.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
        self.model = ViTForImageClassification.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
        self.device = torch.device("mps")
        self.model.to(self.device)
        self.model.eval()

    def __inference(self, tensor):

        with torch.inference_mode():
            outputs = self.model(**tensor)
            logits = outputs.logits

        result = self.model.config.id2label[logits.argmax(-1).item()]
        return result


    @app.post("/predict")
    async def predict_image(self, request: Request, file: UploadFile = File(...)):
        contents = await file.read()  # Read the file contents
        image = Image.open(io.BytesIO(contents))  # Convert bytes to PIL image

        tensor = self.processor(images=image, return_tensors='pt').to(self.device)
        result = self.__inference(tensor)

        prediction_result = {"message":result}
        return JSONResponse(content=prediction_result)


ray.init(address='local',num_gpus=1, num_cpus=1)
ray.serve.start(http_options={'port':5000})
my_app = MedicinalPlantClassification.bind()
ray.serve.run(target=my_app, blocking=True)



