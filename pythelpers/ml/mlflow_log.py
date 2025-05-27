import mlflow
from pythelpers.ml.model_training_analyze import comp_layer_ur

def log_ur(model, step, model_prefix="", lr=None):
    client = mlflow.tracking.MlflowClient()

    metric_listofdict = comp_layer_ur(model, lr=lr)
    # Iterate through named parameters to calculate and log metrics
    for metric in metric_listofdict:
        # Create a formatted name that corresponds to the naming convention in the TensorBoard layout
        formatted_name = f"{model_prefix}{'_' if len(model_prefix) >0 else ''}ur_" + metric['name']
        # Log the metric using the formatted name
        client.log_metric(run_id=mlflow.active_run().info.run_id, key=formatted_name, value=metric['metric'], step=step)
