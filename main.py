import os
import numpy as np


def main():
    print("Welcome to Lydia's Feature Selection Algorithm.\n")

    # Ask user dataset size
    dataset_size = input("Do you want to use the SMALL or LARGE dataset? (type 'small' or 'large'): ").strip().lower()

    if dataset_size == "small":
        folder_path = "./Small_data/"
    elif dataset_size == "large":
        folder_path = "./Large_data/"
    else:
        print("Invalid option. Exiting program.")
        return

    # Ask user for filename
    filename = input("Type in the name of the file to test: ").strip()

    file_path = os.path.join(folder_path, filename)

    if not os.path.exists(file_path):
        print("File not found. Please check the file name and try again.")
        return

    # Load dataset
    data = load_data(file_path)

    num_instances = data.shape[0]
    num_features = data.shape[1] - 1   # first column is class label

    print(f"\nThis dataset has {num_features} features (not including the class attribute), with {num_instances} instances.\n")

    # Ask user which algorithm
    print("Type the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination")

    choice = input("Enter your choice: ").strip()

    if choice == "1":
        forward_selection(data)
    elif choice == "2":
        backward_elimination(data)
    else:
        print("Invalid choice. Exiting.")


def load_data(file_path):
    # Load the dataset file into a numpy array.
    data = np.loadtxt(file_path)
    return data


def forward_selection(data):
    print("\nBeginning Forward Selection search.\n")

    # TODO:
    # 1. Start with empty feature set
    # 2. Iteratively add features
    # 3. Evaluate accuracy with nearest neighbor
    # 4. Track best feature subset

    pass


def backward_elimination(data):
    print("\nBeginning Backward Elimination search.\n")

    # TODO:
    # 1. Start with all features
    # 2. Iteratively remove features
    # 3. Evaluate accuracy with nearest neighbor
    # 4. Track best feature subset

    pass


def nearest_neighbor_accuracy(data, feature_subset):
    # Evaluate accuracy using nearest neighbor with leave-one-out validation.

    # TODO:
    # Implement nearest neighbor classification
    # using only the selected feature subset.

    pass


if __name__ == "__main__":
    main()