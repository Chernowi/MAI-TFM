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

    checkpoint = torch.load(filepath, map_location=device)

    if 'agent_nn_state_dict' in checkpoint:
        agent_nn.load_state_dict(checkpoint['agent_nn_state_dict'])
        print("Loaded agent_nn state dict.")
    else:
        print("Warning: agent_nn_state_dict not found in checkpoint.")

    if 'mixer_nn_state_dict' in checkpoint:
        mixer_nn.load_state_dict(checkpoint['mixer_nn_state_dict'])
        print("Loaded mixer_nn state dict.")
    else:
        print("Warning: mixer_nn_state_dict not found in checkpoint.")

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


if __name__ == '__main__':
    # --- Example Usage & Test ---
    # Dummy networks for testing save/load
    class DummyNet(torch.nn.Module):
        def __init__(self, in_f, out_f):
            super().__init__()
            self.fc = torch.nn.Linear(in_f, out_f)
        def forward(self, x): return self.fc(x)

    agent_net = DummyNet(10, 5)
    mixer_net = DummyNet(5, 1) # Takes num_agents q-values
    
    # Test save
    dummy_state = {
        'episode': 100,
        'best_metric': 0.75,
        'agent_nn_state_dict': agent_net.state_dict(),
        'mixer_nn_state_dict': mixer_net.state_dict(),
    }
    exp_dir = "temp_saved_models/test_exp"
    save_checkpoint(dummy_state, exp_dir, episode=100, metric_value=0.75, is_best=True)

    # Test load
    agent_net_loaded = DummyNet(10, 5)
    mixer_net_loaded = DummyNet(5, 1)
    
    # Find the saved file (or use best_model.pth.tar)
    list_of_files = glob.glob(os.path.join(exp_dir, '*.pth.tar'))
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"\nAttempting to load: {latest_file}")
        loaded_info = load_checkpoint(latest_file, agent_net_loaded, mixer_net_loaded)
        if loaded_info:
            print(f"Loaded episode: {loaded_info.get('episode')}")
            print(f"Loaded best_metric: {loaded_info.get('best_metric')}")
            
            # Verify weights are same (example for one layer)
            assert torch.equal(agent_net.fc.weight, agent_net_loaded.fc.weight)
            print("Weight loading verified for agent_net.")
    else:
        print("No checkpoint files found for loading test.")

    # Clean up
    if os.path.exists(exp_dir):
        for f in glob.glob(os.path.join(exp_dir, "*")): os.remove(f)
        os.rmdir(exp_dir)
        os.rmdir("temp_saved_models")
        print("Cleaned up temp_saved_models.")