import logging
import os
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def setup_logger(experiment_name, log_dir="logs", level=logging.INFO):
    """Sets up a file and console logger."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_subdir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
    if not os.path.exists(log_subdir):
        os.makedirs(log_subdir)
    
    log_filename = os.path.join(log_subdir, "experiment.log")

    logger = logging.getLogger(experiment_name)
    logger.setLevel(level)

    # Prevent multiple handlers if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    fh = logging.FileHandler(log_filename)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    logger.info(f"Logger initialized. Logging to console and {log_filename}")
    return logger, log_subdir # Return subdir for TensorBoard

def get_tensorboard_writer(experiment_name, tb_log_dir="runs"):
    """Initializes a TensorBoard SummaryWriter."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer_log_dir = os.path.join(tb_log_dir, f"{experiment_name}_{timestamp}")
    writer = SummaryWriter(log_dir=writer_log_dir)
    print(f"TensorBoard writer initialized. Logging to: {writer_log_dir}")
    return writer, writer_log_dir

def log_hyperparameters(writer, hparams_dict, metrics_dict=None):
    """Logs hyperparameters and optionally initial metrics to TensorBoard."""
    if metrics_dict is None:
        metrics_dict = {} # Define some default if no initial metrics
    
    # Sanitize hparams for TensorBoard (e.g., convert lists/dicts to strings if too complex)
    sanitized_hparams = {}
    for k, v in hparams_dict.items():
        if isinstance(v, (list, dict)):
            try:
                sanitized_hparams[k] = str(v) 
            except: # If str(v) fails for some reason
                 sanitized_hparams[k] = json.dumps(v,default=str)
        else:
            sanitized_hparams[k] = v
            
    try:
        writer.add_hparams(sanitized_hparams, metrics_dict)
    except Exception as e:
        print(f"Warning: Could not log hparams to TensorBoard: {e}. Hparams: {sanitized_hparams}")
