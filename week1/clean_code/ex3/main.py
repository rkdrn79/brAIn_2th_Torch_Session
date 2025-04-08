# main.py
from data import generate_linear_data
from model import create_linear_model
from train import train_model
from utils import print_model_parameters

def main():
    x, y = generate_linear_data()
    model = create_linear_model()
    model = train_model(model, x, y)
    print_model_parameters(model)

if __name__ == "__main__":
    main()