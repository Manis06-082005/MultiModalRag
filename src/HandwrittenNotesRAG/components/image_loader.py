from pathlib import Path
from PIL import Image

from ..Exception import CustomException
from ..logger import Logger


class ImageLoader:
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def load(self, image_path: str | Path) -> Image.Image:
        image_path = Path(image_path)

        try:
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")

            if image_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"Unsupported image format: {image_path.suffix}"
                )

            with Image.open(image_path) as image:
                loaded_image = image.convert("RGB").copy()

            Logger.info("Loaded image: %s", image_path)
            return loaded_image
        except Exception as error:
            Logger.error("Failed to load image: %s", image_path, exc_info=True)
            raise CustomException(error) from error

    def load_folder(self, folder_path: str | Path) -> list[Image.Image]:
        folder_path = Path(folder_path)

        if not folder_path.is_dir():
            error = NotADirectoryError(f"Folder not found: {folder_path}")
            Logger.error("Invalid image folder: %s", folder_path)
            raise CustomException(error) from error

        images = []
        for image_path in folder_path.iterdir():
            if image_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    images.append(self.load(image_path))
                except Exception as error:
                    Logger.warning(
                        "Skipping image %s: %s",
                        image_path,
                        error,
                        exc_info=True,
                    )

        Logger.info("Loaded %d image(s) from: %s", len(images), folder_path)
        return images
