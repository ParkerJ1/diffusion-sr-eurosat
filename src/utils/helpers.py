import os

def find_latest_model(project_root):
    """
    Find the most recent model checkpoint in the project directory.

    Returns:
        Path to the most recent model.pth file, or None if not found
    """
    import glob

    # Search for all model.pth files in output_* directories
    pattern = os.path.join(project_root, "output_*/model.pth")
    model_files = glob.glob(pattern)

    if not model_files:
        print("No saved models found!")
        return None

    # Sort by modification time (most recent first)
    latest_model = max(model_files, key=os.path.getmtime)

    # Get the timestamp from the directory name
    output_dir = os.path.dirname(latest_model)
    timestamp = os.path.basename(output_dir).replace("output_", "")

    print(f"Found latest model from {timestamp}: {latest_model}")
    return latest_model