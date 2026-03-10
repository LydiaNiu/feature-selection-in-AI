import os
import numpy as np

def forward_selection(data):
    print("\nBeginning Forward Selection search.\n")

    # TODO:
    # 2. Iteratively add features
    # 3. Evaluate accuracy with nearest neighbor
    # 4. Track best feature subset

    # find which class is the majority in the data
    unique, counts = np.unique(data[:,0], return_counts=True)
    majority_count = counts.max()
    baseline_acc = majority_count / len(data[:,0])

    feature_subset = []
    best_acc = baseline_acc
    best_subset = []

    # for loop starts at 1 bc column 0 is class label
    for level in range(1, data.shape[1]):
        best_feature_this_level = None
        best_accuracy_this_level = 0

        for feature in range(1, data.shape[1]):
            if feature not in feature_subset:    
                curr_set = feature_subset.copy()
                curr_set.append(feature)
                temp_acc = calc_accuracy(data, curr_set)
                print(f"    Feature {curr_set} accuracy is {temp_acc:.4f}")

                if (temp_acc >= best_accuracy_this_level):
                    best_feature_this_level = feature
                    best_accuracy_this_level = temp_acc
                
        
        feature_subset.append(best_feature_this_level)
        print(f"Using feature set {feature_subset} was best, accuracy is {best_accuracy_this_level:.4f}\n")
        if best_accuracy_this_level > best_acc:
            best_acc = best_accuracy_this_level
            best_subset = feature_subset.copy()

    print("\n\nFinished search!!")
    print(f"The best feature subset is {best_subset}, which has an accuracy of {best_acc_overall:.4f}")
    

    pass


def backward_elimination(data):
    print("\nBeginning Backward Elimination search.\n")

    # TODO:
    # 1. Start with all features
    # 2. Iteratively remove features
    # 3. Evaluate accuracy with nearest neighbor
    # 4. Track best feature subset

    pass


def calc_accuracy(data, feature_subset):
    # Evaluate accuracy using nearest neighbor with leave-one-out validation.
    
    # 1. create a copy of data to only contain the features/columns from the feature_subset
    data_copy = data[:, feature_subset]
    nearest_dist = 0
    nearest_label = 0
    correct = 0
    # 2. iterate thru all the instances
    for (instance in range(data_copy.shape[0])):

        # 3. for each instance, grab the instance's label
        ins_label = data[instance, 0]   # first column is class label

        # 4. perform nearest neighbor to the rest of the instances (iterate thru the instances again)
        for (k in range(data_copy.shape[0])):   
            if (k != instance): # (k \= i) to skip the instance itself
                # calculate distance from instance to rest of the instances, k
                dist = np.sqrt(np.sum((data_copy[instance] - data_copy[k]) ** 2))
                
                # 5. keep track of the nearest neighbor and its label
                if (dist < nearest_dist):
                    nearest_dist = dist
                    nearest_label = data[k, 0]   # first column is class label

        # 6. compare the nearest neighbor label to the actual label
        if (nearest_label == ins_label):
            correct += 1
   
    # 7. accuracy = correct / total instances
    return correct / data_copy.shape[0]


# find the smallest distance and take its label

# sum( target - neighbor) while the neighbor is just the data itself with only the takes in feature subset 
"""
go thru instaces and then cover 

"""
    # leave one out 

    return accuracy


# 116 76
# 96-98%

def main():
    print("Welcome to Lydia's Feature Selection Algorithm.\n")
    dataset_size = input("Do you want to use the SMALL or LARGE dataset? (type 'small' or 'large'): ").strip().lower()
    # Ask user for filename
    filename = input("Type in the name of the file to test: ").strip()

    if dataset_size == "small":
        folder_path = "./Small_data/"
    elif dataset_size == "large":
        folder_path = "./Large_data/"
    else:
        print("Invalid option. Exiting program.")
        return
    file_path = os.path.join(folder_path, filename)

    if not os.path.exists(file_path):
        print("File not found. Please check the file name and try again.")
        return

    # Load dataset
    data = np.loadtxt(file_path)

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




if __name__ == "__main__":
    main()