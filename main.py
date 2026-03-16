import os
import numpy as np


def forward_selection(data):
    print("\n==============================================\n\nBeginning Forward Selection search.\n")

    # find which class is the majority in the data
    feature_subset = []
    baseline_acc = calc_accuracy(data, feature_subset)
    best_acc = baseline_acc
    best_subset = []
    print(f"Running nearest neighbor with all {data.shape[1]-1} features, using “leaving-one-out” evaluation, the baseline accuracy is {baseline_acc * 100:.2f}%\n\n")
    # for loop starts at 1 bc column 0 is class label
    for level in range(1, data.shape[1]):
        best_feature_this_level = None
        best_accuracy_this_level = 0

        for feature in range(1, data.shape[1]):
            if feature not in feature_subset:    
                curr_set = feature_subset.copy()
                curr_set.append(feature)
                temp_acc = calc_accuracy(data, curr_set)
                print(f"    Using feature {curr_set} accuracy is {temp_acc * 100:.1f}%")

                if temp_acc >= best_accuracy_this_level:
                    best_feature_this_level = feature
                    best_accuracy_this_level = temp_acc
                
        
        feature_subset.append(best_feature_this_level)
        print(f"Feature set {feature_subset} was best, accuracy is {best_accuracy_this_level * 100:.1f}%\n")
        if best_accuracy_this_level > best_acc:
            best_acc = best_accuracy_this_level
            best_subset = feature_subset.copy()

    print("\n\nFinished search!!")
    print(f"\n\nThe best feature subset is {best_subset}, which has an accuracy of {best_acc * 100:.1f}%")
    pass


def backward_elimination(data):
    print("\n==============================================\n\nBeginning Backward Elimination search.\n")

    # find the baseline accuracy using all features
    # note backward elimination starts with all features and removes one at a time, so feature_subset starts with all features
    feature_subset = list(range(1, data.shape[1]))
    baseline_acc = calc_accuracy(data, feature_subset)
    
    # the best accuracy and best subset are the baseline and the copy of the feature subset
    # they will be updated as we go through the levels
    best_acc = baseline_acc
    best_subset = feature_subset.copy()
    
    print(f"Running nearest neighbor with all {data.shape[1]-1} features, using “leaving-one-out” evaluation, the baseline accuracy is {baseline_acc * 100:.2f}%\n\n")
    # for loop starts at data's column 1 bc column 0 is class label
    # unlike forward selection, backward elimination will stop when there's only 1 feature left, so the range is from 1 to data.shape[1]-1
    for level in range(1, data.shape[1]):
        best_accuracy_this_level = 0
        best_feature_to_remove = None
        curr_set = feature_subset.copy()
        for feature in curr_set:
            test_set = curr_set.copy()
            test_set.remove(feature)
            temp_acc = calc_accuracy(data, test_set)
            print(f"    Using feature {test_set} accuracy is {temp_acc * 100:.1f}%")

            if temp_acc >= best_accuracy_this_level:
                best_feature_to_remove = feature
                best_accuracy_this_level = temp_acc
                
        
        feature_subset.remove(best_feature_to_remove)
        print(f"Feature set {feature_subset} was best, accuracy is {best_accuracy_this_level * 100:.1f}%\n")
        if best_accuracy_this_level >= best_acc:
            best_acc = best_accuracy_this_level
            best_subset = feature_subset.copy()

    print("\n\nFinished search!!")
    print(f"\n\nThe best feature subset is {best_subset}, which has an accuracy of {best_acc * 100:.1f}%")

    pass

def calc_accuracy(data, feature_subset):

    # CASE: empty subset → baseline accuracy
    if len(feature_subset) == 0:
        labels = data[:,0]
        counts = np.unique(labels, return_counts=True)[1]
        majority = counts.max()
        return majority / len(labels)

    # Evaluate accuracy using nearest neighbor with leave-one-out validation.
    data_copy = data[:, feature_subset]
    correct = 0

    for instance in range(data_copy.shape[0]):
        nearest_dist = float("inf")
        nearest_label = None
        ins_label = data[instance, 0]

        for k in range(data_copy.shape[0]):
            if k != instance:
                dist = np.sqrt(np.sum((data_copy[instance] - data_copy[k]) ** 2))

                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_label = data[k,0]

        if nearest_label == ins_label:
            correct += 1

    return correct / data_copy.shape[0]


# 116 76
# 96-98%

def main():
    print("\n==============================================\nWelcome to Lydia's Feature Selection Algorithm.\n")
    dataset_size = input("Type your option: \n1) SMALL dataset \n2) LARGE dataset:\n\n")
    if dataset_size == "1":
        folder_path = "./Small_data/"
        prefix = "CS170_Small_DataSet__"
    elif dataset_size == "2":
        folder_path = "./Large_data/"
        prefix = "CS170_Large_DataSet__"
    elif dataset_size == "3":
        folder_path = "./SanityCheck/"
        prefix = "SanityCheck_DataSet__"
    else:
        print("Invalid option. Exiting program.")
        return
    
    # Ask user for filename
    dataset_number = input("\nPlease enter the dataset number: \n(My assigned testing datasets are SMALL 116 && LARGE 76)\n\n").strip()    
    filename = f"{prefix}{dataset_number}.txt"
    file_path = os.path.join(folder_path, filename)


    if not os.path.exists(file_path):
        print("File not found. Please check the file name and try again.")
        return

    # Load dataset
    data = np.loadtxt(file_path)

    num_instances = data.shape[0]
    num_features = data.shape[1] - 1   # first column is class label

    print(f"\nGreat choice! We will be working on {filename}")
    print(f"\nThis dataset has {num_features} features (not including the class attribute), with {num_instances} instances.\n")

    # Ask user which algorithm
    print("Which algorithm would you like to run? (Please type 1 or 2)")
    print("1) Forward Selection")
    print("2) Backward Elimination")

    choice = input("\nEnter your choice: ").strip()

    if choice == "1":
        forward_selection(data)
    elif choice == "2":
        backward_elimination(data)
    else:
        print("Invalid choice. Exiting.")




if __name__ == "__main__":
    main()