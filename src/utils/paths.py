from pathlib import Path


def get_output_dir(subdir: str = "") -> Path:
    """
    Get the output directory path, creating it if it doesn't exist.
    
    The output directory is located at the root of the backend project.
    Subdirectories can be specified for organization (e.g., 'synthetic_vae_data').
    
    Args:
        subdir: Optional subdirectory within outputs/ (e.g., 'synthetic_vae_data')
        
    Returns:
        Path object pointing to the output directory
        
    Example:
        >>> output_dir = get_output_dir('synthetic_vae_data')
        >>> # Returns: /path/to/backend/outputs/synthetic_vae_data/
    """
    # Get the backend root directory (where this file's parent's parent's parent is)
    backend_root = Path(__file__).parent.parent.parent
    
    # Create the outputs directory path
    if subdir:
        output_path = backend_root / "outputs" / subdir
    else:
        output_path = backend_root / "outputs"
    
    # Create the directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    return output_path

