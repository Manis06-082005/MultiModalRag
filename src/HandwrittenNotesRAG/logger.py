import os 
import logging 
from datetime import datetime
log_dir="logs"
os.makedirs(log_dir,exist_ok=True)
log_file=f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log"
log_file_path=os.path.join(log_dir,log_file)
logging.basicConfig(
    filename=log_file_path,
    format="[ %(asctime)s ] %(levelname)s - %(message)s",
    level=logging.INFO
)
Logger=logging.getLogger(__file__)