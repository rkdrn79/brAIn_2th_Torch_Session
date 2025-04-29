from data.data import load_data
from model import create_model
from train import train_model

def main():
    train_loader = load_data()
    model = create_model()
    train_model(model, train_loader)


if __name__ == "__main__":
    main()