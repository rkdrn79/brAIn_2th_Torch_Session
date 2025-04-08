# utils.py
def print_model_parameters(model):
    print("Weight:", model.weight.data)
    print("Bias:", model.bias.data)