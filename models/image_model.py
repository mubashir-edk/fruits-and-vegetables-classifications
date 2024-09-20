from pydantic import BaseModel, HttpUrl

class ImageData(BaseModel):
    image_url: HttpUrl