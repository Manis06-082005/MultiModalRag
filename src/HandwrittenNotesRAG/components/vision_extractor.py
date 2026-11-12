
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from ..Exception import CustomException
from ..logger import Logger


class VisionExtractor:

    MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

    def __init__(self) -> None:

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.MODEL_ID
            )

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.MODEL_ID,
                dtype="auto",
                device_map="auto"
            )

            Logger.info("Vision model initialized successfully")

        except Exception as error:
            Logger.error("Vision model initialization failed", exc_info=True)
            raise CustomException(error) from error

    def extract_description(self, image: Image.Image) -> str:

        try:
            image = image.convert("RGB")

            prompt = """
            Analyze this educational image.

            Describe:
            1. What the image represents.
            2. Important objects, labels and components.
            3. How the components are connected.
            4. The process or concept being explained.

            Explain clearly and do not invent information
            that is not visible in the image.
            """

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device)

            with torch.inference_mode():

                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False
                )

            generated_ids_trimmed = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(
                    inputs["input_ids"],
                    generated_ids
                )
            ]

            description = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True
            )[0].strip()

            Logger.info("Image description generated successfully")

            return description

        except Exception as error:
            Logger.error("Vision extraction failed", exc_info=True)
            raise CustomException(error) from error
