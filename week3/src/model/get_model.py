def get_model(args):
    """
    Function to get the model based on the provided arguments.
    
    Args:
        args: Command line arguments containing model type and other configurations.
    
    Returns:
        model: The initialized model.
    """
    if args.model_type == "CNN":
        from src.model.cnn import CNN
        model = CNN(input_dim=args.input_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim)

    return model