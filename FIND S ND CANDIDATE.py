import csv

# Function to read training data from a CSV file
def read_training_data(filename):
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        training_data = list(csv_reader)
    return training_data

# FIND-S Algorithm for finding the most specific hypothesis
def find_s_algorithm(training_data):
    hypothesis = training_data[0][:-1]  # Initialize with the first instance
    for instance in training_data:
        if instance[-1] == 'Yes':  # Positive instance
            for i in range(len(hypothesis)):
                if instance[i] != hypothesis[i]:
                    hypothesis[i] = '?'  # Generalize the hypothesis
    return hypothesis

# Candidate-Elimination Algorithm
def candidate_elimination(training_data):
    specific_hypothesis = training_data[0][:-1]  # Initialize with the first instance
    general_hypothesis = [['?'] * (len(specific_hypothesis))]

    for instance in training_data:
        if instance[-1] == 'Yes':  # Positive instance
            for i in range(len(specific_hypothesis)):
                if instance[i] != specific_hypothesis[i]:
                    specific_hypothesis[i] = '?'  # Generalize the specific hypothesis
            new_general_hypotheses = []
            for general in general_hypothesis:
                for i in range(len(general)):
                    if general[i] != '?' and general[i] != specific_hypothesis[i]:
                        new_general = list(general)
                        new_general[i] = '?'
                        if new_general not in general_hypothesis:
                            new_general_hypotheses.append(new_general)
                general_hypothesis.extend(new_general_hypotheses)
                general_hypothesis = [h for h in general_hypothesis if h != ['?'] * len(specific_hypothesis)]
        else:  # Negative instance
            new_general_hypotheses = []
            for general in general_hypothesis:
                if instance[:-1] != general:
                    new_general_hypotheses.append(general)
            general_hypothesis = new_general_hypotheses

    return specific_hypothesis, general_hypothesis

# Main function
def main():
    # Read training data from CSV file
    training_data = read_training_data(r'C:\Users\Admin\Downloads\bharthdataset.csv')

    # Display training data
    print("Training Data:")
    for row in training_data:
        print(row)

    # FIND-S Algorithm
    print("\nFIND-S Algorithm:")
    find_s_hypothesis = find_s_algorithm(training_data)
    print("Most Specific Hypothesis:", find_s_hypothesis)

    # Candidate-Elimination Algorithm
    print("\nCandidate-Elimination Algorithm:")
    ce_specific_hypothesis, ce_general_hypothesis = candidate_elimination(training_data)
    print("Most Specific Hypothesis:", ce_specific_hypothesis)
    print("General Hypotheses:")
    for general in ce_general_hypothesis:
        print(general)

if __name__ == "__main__":
    main()
