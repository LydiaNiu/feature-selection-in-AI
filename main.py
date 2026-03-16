import os
import numpy as np

"""
Forward Selection and Backward Elimination are two greedy algorithms for feature selection.

Forward Selection: It starts with an empty feature subset. Each iteration, the algorithm adds one feature 
                   at a time that gives the best accuracy until all features are added. 
Backward Elimination: It starts with all features. Each iteration, the algorithm removes one feature at a time 
                   that gives the best accuracy until there are no features left.
In both algorithms, we use leave-one-out validation to evaluate the accuracy of the current feature subset.
"""

def forward_selection(data):
    print("\n==============================================\n\nBeginning Forward Selection search.\n")

    # find the baseline accuracy using 0 features, majority class's accuracy / total instances
    # initialized the variables that will keep track of the best accuracy and the corresponding feature subset
    feature_subset = []
    baseline_acc = calc_accuracy(data, feature_subset)
    best_acc = baseline_acc
    best_subset = []
    print(f"Running nearest neighbor with all {data.shape[1]-1} features, using “leaving-one-out” evaluation, the baseline accuracy is {baseline_acc * 100:.2f}%\n\n")
    
    # I use a while loop here that continues until all features are added
    while len(feature_subset) < data.shape[1] - 1:
        # each level, the best feature and the accuracy will be updated
        best_feature_this_level = None
        best_accuracy_this_level = -1 # initialized as -1 so all the accuracies will be better than it
        curr_set = feature_subset.copy() # create a copy to prevent from modifying the original feature subset

        # a for loop to iterate through all the features that are not in the current feature subset, 
        # and find the best feature to add by comparison
        for feature in range(1, data.shape[1]):
            if feature not in curr_set:
                test_set = curr_set.copy()
                test_set.append(feature)
                temp_acc = calc_accuracy(data, test_set)
                print(f"    Using feature {test_set} accuracy is {temp_acc * 100:.1f}%")
                
                # keep track of the best accuracy and the corresponding feature by comparison
                if temp_acc > best_accuracy_this_level:
                    best_feature_this_level = feature
                    best_accuracy_this_level = temp_acc
                
        # the best feature of this level is added to the feature subset, and we print the accuracy of the new subset
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

    # note backward elimination starts with all features and removes one at a time, 
    # so baseline accuracy is calculated using all features, 
    # and feature_subset starts with all features
    feature_subset = list(range(1, data.shape[1]))
    baseline_acc = calc_accuracy(data, feature_subset)
    best_acc = baseline_acc
    best_subset = feature_subset.copy()
    print(f"Running nearest neighbor with all {data.shape[1]-1} features, using “leaving-one-out” evaluation, the baseline accuracy is {baseline_acc * 100:.2f}%\n\n")
    
    while len(feature_subset) > 0:
        best_accuracy_this_level = -1
        best_feature_to_remove = None
        curr_set = feature_subset.copy()
        
        for feature in curr_set:
            test_set = curr_set.copy()
            test_set.remove(feature) # instead of append, we remove
            temp_acc = calc_accuracy(data, test_set)
            print(f"    Using feature {test_set} accuracy is {temp_acc * 100:.1f}%")

            if temp_acc > best_accuracy_this_level:
                best_feature_to_remove = feature
                best_accuracy_this_level = temp_acc
                
        
        feature_subset.remove(best_feature_to_remove)
        print(f"Feature set {feature_subset} was best, accuracy is {best_accuracy_this_level * 100:.1f}%\n")
        if best_accuracy_this_level > best_acc:
            best_acc = best_accuracy_this_level
            best_subset = feature_subset.copy()

    print("\n\nFinished search!!")
    print(f"\n\nThe best feature subset is {best_subset}, which has an accuracy of {best_acc * 100:.1f}%")

    pass

"""
This function calculates the accuracy of the dataset that only contains the features in the given subset
using leave-one-out validation. There are two cases to consider:
1) If the feature subset is empty, we return the baseline accuracy with 0 features (majority class accuracy).
2) If there are features in the subset, calculate the distance from each instance to the rest instances (k)
Then predict the label of the instance based on the nearest neighbor's label and compare it to the actual label.
If the predicted label matches the actual label, we increment the correct count.
"""
def calc_accuracy(data, feature_subset):

    if len(feature_subset) == 0:
        labels = data[:,0]
        counts = np.unique(labels, return_counts=True)[1]
        return counts.max() / len(labels)

    features = data[:, feature_subset]
    correct = 0
    n = data.shape[0]

    # Here, i is the level of the algorithm. Each level will find the best feature to remove from 
    # the current feature subset. 
    # The algorithm will continue until there are no more features left in the subset.
    for i in range(n):
        
        # In stead of looping through to calculate the distance between the i-th instance and the rest of the instances, 
        # we can use vectorized operations to calculate the distances more efficiently.
        # I consult that this method is advised from AI tool to optimize the runtime, but all the codes are written by me.
        # The original codes are commented out below for reference.
        diff = features - features[i] 
        dists = np.sum(diff**2, axis=1) # the usage of sum() instead of sqrt() is sufficient because the ordering stays the same for nearest neighbor.

        dists[i] = float("inf")

        nearest_index = np.argmin(dists)

        # ========original version============
        # nearest_dist = float("inf")
        # nearest_label = None
        # ins_label = data[instance, 0]

        # for k in range(data.shape[0]):
        #     if k != instance:
        #         diff = data[instance, feature_subset] - data[k, feature_subset]
        #         dist = np.sum(diff ** 2)
        #         if dist < nearest_dist:
        #             nearest_dist = dist
        #             nearest_label = data[k,0]
        
        if data[nearest_index,0] == data[i,0]:
            correct += 1
        
    return correct / n

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