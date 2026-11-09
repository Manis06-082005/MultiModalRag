from PIL import Image,ImageOps

from ..Exception import CustomException
from ..logger import Logger

class ImagePreprocessor:
    def __init__(self,max_size:tuple[int,int] | None=(2000,2000))-> None:
        self.max_size=max_size
    def preprocess(self,image:Image.Image)->Image.Image:
        try:
            processed_image=ImageOps.grayscale(image)
            processed_image=ImageOps.autocontrast(processed_image)
            
            if self.max_size is not None:
                processed_image.thumbnail(
                    self.max_size,
                    Image.Resampling.LANCZOS
                )
            Logger.info("Preprocessed image ")
            return processed_image
        except Exception as error:
            Logger.error("failed to preprocess image ",exc_info=True)
            raise CustomException(error) from error