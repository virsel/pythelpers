import os
import logging
from torch import nn
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter



def set_logging():
    # set logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
    # Disable c10d logging
    logging.getLogger('c10d').setLevel(logging.ERROR)
    
    
class TensorLogger:
    def __init__(self, logdir='./logs'):
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logdir = os.path.join(logdir, current_datetime)
        # Ensure the log directory exists
        os.makedirs(logdir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=logdir)
        
    def log_metric(self, value, step, metric='loss', mode='train'):
        self.writer.add_scalar(f'{mode}/{metric}', value, step)
        
    def log_report(self, report, step, mode='val'):
        # Log metrics from the classification report
        for label, metrics in report.items():
            # Check if metrics is a dictionary
            if isinstance(metrics, dict):  # This will be True for class labels and averages
                for metric_name, value in metrics.items():
                    # Log class-specific metrics and averages except 'support'
                    if metric_name != 'support':
                        self.writer.add_scalar(f'{mode}_class_{label}/{metric_name}', value, step)
            else:
                # This handles the overall 'accuracy', which is a single float value
                # Log overall accuracy
                if label == 'accuracy':
                    self.writer.add_scalar(f'{mode}/{label}', metrics, step)
    
    def log_text(self, txt, step):
        # Log some text
        self.writer.add_text('Version Description', txt, step)
        self.writer.flush()
        
    def log_ud(self, model, step, model_prefix="", lr=None):
        # Get the current learning rate
        if lr is None:
            lr = model.optimizers().param_groups[0]['lr']

        # Iterate through named parameters to calculate and log metrics
        for name, p in model.named_parameters():
            if p.ndim == 2 and p.grad is not None:
                # Calculate the standard deviation of the gradients adjusted by the learning rate
                grad_std = (lr * p.grad).std()
                # Calculate the standard deviation of the parameter values
                param_std = p.data.std()
                # Calculate the Update Discrepancy (ud) metric and take the log10
                metric = (grad_std / param_std).log10().item()
                # Create a formatted name that corresponds to the naming convention in the TensorBoard layout
                formatted_name = f"{model_prefix}{'_' if len(model_prefix) >0 else ''}z_ud_" + name.replace('.', '_')
                # Log the metric using the formatted name
                self.log_metric(metric, step, formatted_name)
                
    def log_model_arch(self, model):
        # Log the model architecture at the start of training
        model_info = "<br>".join([f"{name}: {str(el)}" for name, el in model.get_elements().items()])
        self.writer.add_text('Model Architecture', model_info, 0)
