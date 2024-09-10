from fastapi import APIRouter
from fastapi import UploadFile, File
from app.utils import predict_image_class
from PIL import Image
from io import BytesIO
from app.schema import prediction

router = APIRouter(prefix="/api")

@router.post("/classify", response_model=prediction)
async def classify(image: UploadFile = File(...)):
    '''Image is posted at classify and the class name and confidence score is returned'''
    file = await image.read()
    image = Image.open(BytesIO(file)).convert("RGB")
    class_name, confidence_score = await predict_image_class(image)
    print("Class Name: ", class_name)
    print("Confidence Score: ", confidence_score)
    return {"class_name": class_name, "confidence_score": confidence_score}

