import torch
import os
import glob

def save_checkpoint(state, experiment_dir, filename_prefix="model", episode=None, metric_value=None, is_best=False):
    """
    Saves model and optimizer state.
    Args:
        state (dict): Contains 'agent_nn_state_dict', 'mixer_nn_state_dict', 
                      'agent_target_nn_state_dict', 'mixer_target_nn_state_dict',
                      'optimizer_state_dict' (optional), etc.
        experiment_dir (str): Directory to save the checkpoint.
        filename_prefix (str): Prefix for the filename.
        episode (int, optional): Current episode number.
        metric_value (float, optional): Metric value (e.g., IoU) for naming.
        is_best (bool): If True, also save as 'best_model.pth.tar'.
    """
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    filename_parts = [filename_prefix]
    if episode is not None:
        filename_parts.append(f"ep{episode}")
    if metric_value is not None:
        filename_parts.append(f"metric{metric_value:.3f}".replace('.', 'p')) # Replace . with p for filename
    
    filename = "_".join(filename_parts) + ".pth.tar"
    filepath = os.path.join(experiment_dir, filename)
    
    torch.save(state, filepath)
    print(f"Saved checkpoint: {filepath}")

    if is_best:
        best_filepath = os.path.join(experiment_dir, "best_model.pth.tar")
        torch.save(state, best_filepath)
        print(f"Saved best model checkpoint: {best_filepath}")


def load_checkpoint(filepath, agent_nn, mixer_nn, agent_target_nn=None, mixer_target_nn=None, optimizer=None, device='cpu'):
    """
    Loads model and optimizer state from a checkpoint.
    Args:
        filepath (str): Path to the checkpoint file.
        agent_nn (torch.nn.Module): Agent policy network instance.
        mixer_nn (torch.nn.Module): Mixer network instance.
        agent_target_nn (torch.nn.Module, optional): Agent target network.
        mixer_target_nn (torch.nn.Module, optional): Mixer target network.
        optimizer (torch.optim.Optimizer, optional): Optimizer instance.
        device (str): Device to load tensors to.
    Returns:
        dict: The loaded state dictionary (e.g., for resuming with epoch, best_metric).
    """
    if not os.path.exists(filepath):
        print(f"Checkpoint file not found: {filepath}")
        return None

    # This is the line that needs to be changed.
    # We add `weights_only=False` to allow loading older checkpoints
    # that may contain non-tensor Python objects like numpy scalars.
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    if 'agent_nn_state_dict' in checkpoint:
        agent_nn.load_state_dict(checkpoint['agent_nn_state_dict'])
        print("Loaded agent_nn state dict.")
    else:
        print("Warning: agent_nn_state_dict not found in checkpoint.")

    if mixer_nn and 'mixer_nn_state_dict' in checkpoint:
        mixer_nn.load_state_dict(checkpoint['mixer_nn_state_dict'])
        print("Loaded mixer_nn state dict.")
    elif mixer_nn:
        print("Warning: mixer_nn_state_dict not found, but mixer network was provided.")

    if agent_target_nn and 'agent_target_nn_state_dict' in checkpoint:
        agent_target_nn.load_state_dict(checkpoint['agent_target_nn_state_dict'])
        print("Loaded agent_target_nn state dict.")
    elif agent_target_nn:
        print("Warning: agent_target_nn_state_dict not found, but target network provided.")
        
    if mixer_target_nn and 'mixer_target_nn_state_dict' in checkpoint:
        mixer_target_nn.load_state_dict(checkpoint['mixer_target_nn_state_dict'])
        print("Loaded mixer_target_nn state dict.")
    elif mixer_target_nn:
         print("Warning: mixer_target_nn_state_dict not found, but target network provided.")

    if optimizer and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded optimizer state dict.")
        except Exception as e:
            print(f"Warning: Could not load optimizer state dict: {e}. Optimizer might be reset.")
    elif optimizer:
        print("Warning: optimizer_state_dict not found, but optimizer provided.")
    
    print(f"Loaded checkpoint from {filepath}")
    return checkpoint # Contains other info like epoch, best_metric etc.