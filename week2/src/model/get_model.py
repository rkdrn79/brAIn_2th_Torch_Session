def get_model(args):
    """
    Function to get the model based on the provided arguments.
    
    Args:
        args: Command line arguments containing model type and other configurations.
    
    Returns:
        model: The initialized model.
    """
    if args.model_type == "MLP":
        from src.model.mlp import MLPModel
        model = MLPModel(args)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    return model 