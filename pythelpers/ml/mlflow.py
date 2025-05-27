from typing import Dict, Any, Optional
import torch
from zenml.logger import get_logger # ZenML's logger
import mlflow
import mlflow.tracking # For MlflowClient
from pathlib import Path
import tempfile # For temporary file handling
import re # For parsing epoch from filename
import os # For file operations



logger = get_logger(__name__) # Use ZenML's logger for the step



# --- Helper Function to Load Full Resumable Checkpoint from MLflow ---
def load_latest_checkpoint2(
    run_id: str, # Current MLflow run ID
    artifact_subdir: str, # e.g., "manual_model_checkpoints"
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, # type: ignore
    device: str = "cpu"
) -> Optional[Dict[str, Any]]:
    """
    Loads the latest full resumable checkpoint from a subdirectory in MLflow artifacts
    for the current run.
    Sorts checkpoints by epoch number parsed from filenames.
    """
    client = mlflow.tracking.MlflowClient()
    try:
        artifacts = client.list_artifacts(run_id=run_id, path=artifact_subdir)
    except Exception as e:
        logger.warning(f"Could not list artifacts in '{artifact_subdir}' for run_id '{run_id}': {e}. Assuming no checkpoint.")
        return None

    checkpoints = []
    for art in artifacts:
        if art.is_dir:
            continue
        # Expecting filenames like model_checkpoint_epoch_0042.pt or model_epoch_42_val_1.23.pt
        match = re.search(r"epoch_(\d+)", art.path)
        if match:
            epoch = int(match.group(1))
            checkpoints.append({"epoch": epoch, "path": art.path, "artifact_full_path": art.path})

    if not checkpoints:
        logger.info(f"No checkpoints found in MLflow artifacts path '{artifact_subdir}'.")
        return None

    # Sort by epoch descending to get the latest
    checkpoints.sort(key=lambda x: x["epoch"], reverse=True)
    latest_checkpoint_info = checkpoints[0]
    
    logger.info(f"Found latest checkpoint: {latest_checkpoint_info['path']} (epoch {latest_checkpoint_info['epoch']})")

    try:
        with tempfile.TemporaryDirectory() as tmp_download_dir:
            downloaded_artifact_path_str = client.download_artifacts(
                run_id=run_id,
                path=latest_checkpoint_info["artifact_full_path"], # Path relative to artifact root
                dst_path=tmp_download_dir
            )
            actual_checkpoint_file = Path(downloaded_artifact_path_str)
            
            if actual_checkpoint_file.exists() and actual_checkpoint_file.is_file():
                logger.info(f"Loading checkpoint from '{actual_checkpoint_file}'...")
                checkpoint_data = torch.load(actual_checkpoint_file, map_location=torch.device(device))

                model.load_state_dict(checkpoint_data['model_state_dict'])
                if optimizer and 'optimizer_state_dict' in checkpoint_data:
                    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                if scheduler and 'scheduler_state_dict' in checkpoint_data:
                    scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                
                logger.info(f"Resuming training. Loaded state from epoch: {checkpoint_data.get('epoch', -1)}")
                return checkpoint_data # Return the whole dict
            else:
                logger.error(f"Downloaded artifact path '{actual_checkpoint_file}' is not a valid file.")
                return None
    except Exception as e:
        logger.error(f"Error loading checkpoint {latest_checkpoint_info['path']}: {e}")
        return None
    
# --- Helper Function to Load Full Resumable Checkpoint from MLflow ---
def load_latest_checkpoint(
    run_id: str, # Current MLflow run ID
    artifact_subdir: str, # e.g., "manual_model_checkpoints"
    device: str = "cpu"
) -> Optional[Dict[str, Any]]:
    """
    Loads the latest full resumable checkpoint from a subdirectory in MLflow artifacts
    for the current run.
    Sorts checkpoints by epoch number parsed from filenames.
    """
    client = mlflow.tracking.MlflowClient()
    try:
        artifacts = client.list_artifacts(run_id=run_id, path=artifact_subdir)
    except Exception as e:
        logger.warning(f"Could not list artifacts in '{artifact_subdir}' for run_id '{run_id}': {e}. Assuming no checkpoint.")
        return None

    checkpoints = []
    for art in artifacts:
        if art.is_dir:
            continue
        # Expecting filenames like model_checkpoint_epoch_0042.pt or model_epoch_42_val_1.23.pt
        match = re.search(r"epoch_(\d+)", art.path)
        if match:
            epoch = int(match.group(1))
            checkpoints.append({"epoch": epoch, "path": art.path, "artifact_full_path": art.path})

    if not checkpoints:
        logger.info(f"No checkpoints found in MLflow artifacts path '{artifact_subdir}'.")
        return None

    # Sort by epoch descending to get the latest
    checkpoints.sort(key=lambda x: x["epoch"], reverse=True)
    latest_checkpoint_info = checkpoints[0]
    
    logger.info(f"Found latest checkpoint: {latest_checkpoint_info['path']} (epoch {latest_checkpoint_info['epoch']})")

    try:
        with tempfile.TemporaryDirectory() as tmp_download_dir:
            downloaded_artifact_path_str = client.download_artifacts(
                run_id=run_id,
                path=latest_checkpoint_info["artifact_full_path"], # Path relative to artifact root
                dst_path=tmp_download_dir
            )
            actual_checkpoint_file = Path(downloaded_artifact_path_str)
            
            if actual_checkpoint_file.exists() and actual_checkpoint_file.is_file():
                logger.info(f"Loading checkpoint from '{actual_checkpoint_file}'...")
                checkpoint_data = torch.load(actual_checkpoint_file, map_location=torch.device(device))
        
                logger.info(f"Resuming training. Loaded state from epoch: {checkpoint_data.get('epoch', -1)}")
                return checkpoint_data # Return the whole dict
            else:
                logger.error(f"Downloaded artifact path '{actual_checkpoint_file}' is not a valid file.")
                return None
    except Exception as e:
        logger.error(f"Error loading checkpoint {latest_checkpoint_info['path']}: {e}")
        return None

# --- Helper function for checkpoint rotation ---
def rotate_checkpoints(
    run_id: str,
    artifact_subdir: str,
    max_checkpoints: int
):
    """Keeps only the 'max_checkpoints' most recent checkpoints in the artifact_subdir."""
    client = mlflow.tracking.MlflowClient()
    try:
        artifacts = client.list_artifacts(run_id=run_id, path=artifact_subdir)
    except Exception as e:
        logger.warning(f"Could not list artifacts for rotation in '{artifact_subdir}': {e}")
        return

    checkpoints = []
    for art in artifacts:
        if art.is_dir:
            continue
        match = re.search(r"epoch_(\d+)", art.path) # Assumes epoch is in filename
        if match:
            epoch = int(match.group(1))
            checkpoints.append({"epoch": epoch, "path": art.path}) # art.path is relative to run artifact root

    if len(checkpoints) <= max_checkpoints:
        return

    # Sort by epoch ascending to find the oldest
    checkpoints.sort(key=lambda x: x["epoch"])
    
    num_to_delete = len(checkpoints) - max_checkpoints
    checkpoints_to_delete = checkpoints[:num_to_delete]

    for ckpt_info in checkpoints_to_delete:
        try:
            logger.info(f"Deleting old checkpoint: {ckpt_info['path']}")
            delete_artifact_local(run_id, ckpt_info['path'])
        except Exception as e:
            logger.error(f"Failed to delete old checkpoint {ckpt_info['path']}: {e}")

def rotate_bestmodels(
    run_id: str,
    metric: str = "loss",
    artifact_subdir: str = "",
    max: int = 5,
    maximize: bool = False
):
    """
    Keeps only the 'max' best checkpoints in the artifact_subdir, 
    based on the value of the specified metric in the checkpoint filename.
    By default, keeps the checkpoints with the lowest metric (assumed to be loss).
    Set maximize=True to keep the highest values (e.g., for accuracy).
    """
    client = mlflow.tracking.MlflowClient()
    try:
        artifacts = client.list_artifacts(run_id=run_id, path=artifact_subdir)
    except Exception as e:
        logger.warning(f"Could not list artifacts for rotation in '{artifact_subdir}': {e}")
        return

    checkpoints = []
    # Assumes checkpoint filenames contain the metric, e.g., "bestmodel_loss_0001.pt" or "model_valacc_0.9421.pt"
    metric_regex = re.compile(rf"{re.escape(metric)}[_\-]?([0-9]+(?:\.[0-9]+)?)")

    for art in artifacts:
        if art.is_dir:
            continue
        match = metric_regex.search(art.path)
        if match:
            try:
                metric_val = float(match.group(1))
            except ValueError:
                continue
            checkpoints.append({"metric": metric_val, "path": art.path})

    if len(checkpoints) <= max:
        return

    # Sort by metric ascending (minimize) or descending (maximize)
    checkpoints.sort(key=lambda x: x["metric"], reverse=maximize)

    # Keep the best 'max', delete the rest
    to_delete = checkpoints[max:]
    for ckpt_info in to_delete:
        try:
            logger.info(f"Deleting old best model (metric={metric}, value={ckpt_info['metric']}): {ckpt_info['path']}")
            delete_artifact_local(run_id, ckpt_info['path'])
        except Exception as e:
            logger.error(f"Failed to delete old best model {ckpt_info['path']}: {e}")

def load_best_model(
    run_id: str,
    artifact_subdir: str = "",
    metric: str = "loss",
    maximize: bool = False,
    device: str = "cpu"
) -> Optional[Dict[str, Any]]:
    """
    Loads the best checkpoint from a subdirectory in MLflow artifacts for the given run,
    based on the value of the specified metric in the checkpoint filename.
    By default, returns the checkpoint with the lowest metric (e.g., for loss).
    Set maximize=True to load the highest value (e.g., for accuracy).
    """
    client = mlflow.tracking.MlflowClient()
    try:
        artifacts = client.list_artifacts(run_id=run_id, path=artifact_subdir)
    except Exception as e:
        logger.warning(f"Could not list artifacts in '{artifact_subdir}' for run_id '{run_id}': {e}.")
        return None

    # Assumes filenames like "bestmodel_loss_0.1234.pt" or "model_valacc_0.9421.pt"
    metric_regex = re.compile(rf"{re.escape(metric)}[_\-]?([0-9]+(?:\.[0-9]+)?)")
    checkpoints = []
    for art in artifacts:
        if art.is_dir:
            continue
        match = metric_regex.search(art.path)
        if match:
            try:
                metric_val = float(match.group(1))
            except ValueError:
                continue
            checkpoints.append({"metric": metric_val, "path": art.path})

    if not checkpoints:
        logger.info(f"No checkpoints with metric '{metric}' found in MLflow artifacts path '{artifact_subdir}'.")
        return None

    # Sort: ascending (minimize) by default, descending if maximize
    checkpoints.sort(key=lambda x: x["metric"], reverse=maximize)
    best_ckpt = checkpoints[0]

    logger.info(f"Found best checkpoint: {best_ckpt['path']} ({metric}={best_ckpt['metric']})")
    try:
        with tempfile.TemporaryDirectory() as tmp_download_dir:
            downloaded_path = client.download_artifacts(
                run_id=run_id,
                path=best_ckpt['path'],
                dst_path=tmp_download_dir
            )
            actual_ckpt_file = Path(downloaded_path)
            if actual_ckpt_file.exists() and actual_ckpt_file.is_file():
                logger.info(f"Loading checkpoint from '{actual_ckpt_file}'...")
                checkpoint_data = torch.load(actual_ckpt_file, map_location=torch.device(device))
                logger.info(f"Loaded checkpoint with {metric}: {best_ckpt['metric']}")
                return checkpoint_data
            else:
                logger.error(f"Downloaded artifact path '{actual_ckpt_file}' is not a valid file.")
                return None
    except Exception as e:
        logger.error(f"Error loading best checkpoint {best_ckpt['path']}: {e}")
        return None



def delete_artifact_local(run_id, artifact_path):
    client = mlflow.tracking.MlflowClient()
    local_path = client.download_artifacts(run_id, artifact_path)
    if os.path.isfile(local_path):
        os.remove(local_path)
        logger.info(f"Deleted local artifact: {local_path}")
    else:
        logger.warning(f"File not found for deletion: {local_path}")

# --- Helper function for checkpoint rotation ---
def remove_all_from_artifact_dir(
    run_id: str,
    manual_bestmodel_subdir: str,
):
    try:
        client = mlflow.tracking.MlflowClient()
        artifacts_in_subdir = client.list_artifacts(run_id=run_id, path=manual_bestmodel_subdir)
        for old_artifact in artifacts_in_subdir:
            if not old_artifact.is_dir: # Ensure it's a file
                logger.info(f"Deleting previous best model artifact from MLflow: {old_artifact.path}")
                client.delete_artifact(run_id=run_id, artifact_path=old_artifact.path)
    except Exception as e:
        # Log a warning if listing/deleting fails, but proceed to save the new one.
        # This could happen if the subdir doesn't exist yet (first best model).
        logger.warning(f"Could not list/delete old best model artifacts from '{manual_bestmodel_subdir}': {e}. This is normal if no best model was previously saved there.")
