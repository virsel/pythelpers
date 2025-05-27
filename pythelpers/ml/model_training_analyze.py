def comp_layer_ur(model, lr=None):
    # compute layer update grad relative to the parameter std
    if lr is None:
        lr = model.optimizers().param_groups[0]['lr']

    metric_listofdict = []
    # Iterate through named parameters to calculate and log metrics
    for name, p in model.named_parameters():
        if p.ndim == 2 and p.grad is not None:
            # Calculate the standard deviation of the gradients adjusted by the learning rate
            grad_std = (lr * p.grad).std()
            # Calculate the standard deviation of the parameter values
            param_std = p.data.std()
            # Calculate the Update Discrepancy (ud) metric and take the log10
            metric = (grad_std / param_std).log10().item()
            metric_listofdict.append({'name': name.replace('.', '_'), 'metric': metric})
    
    return metric_listofdict
            