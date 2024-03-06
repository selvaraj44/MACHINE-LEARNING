import csv

# Initialize the version space with the most general and specific hypotheses
def initialize_hypotheses(data):
    num_attributes = len(data[0]) - 1
    specific_hypothesis = ['0'] * num_attributes
    general_hypothesis = ['?'] * num_attributes
    return specific_hypothesis, general_hypothesis

def is_consistent(instance, hypothesis):
    for i in range(len(instance) - 1):  # Exclude the last column (class label)
        if hypothesis[i] != '?' and instance[i] != hypothesis[i]:
            return False



    return True

# Generalize the specific hypothesis
def generalize(specific, instance):
    for i in range(len(specific)):
        if specific[i] == '0':
            specific[i] = instance[i]
        elif specific[i] != instance[i]:
            specific[i] = '?'
    return specific


# Specialize the general hypothesis
def specialize(general, instance, specific):
    new_general = [h for h in general]
    for i in range(len(general)):
        if general[i] == '?':
            if instance[i] != specific[i]:
                new_general[i] = specific[i]
            else:
                new_general[i] = '?'
    return new_general

# Candidate-Elimination algorithm
def candidate_elimination(data):
    specific_hypothesis, general_hypothesis = initialize_hypotheses(data)
    for instance in data:
        if instance[-1] == 'Yes':  # Positive instance
            specific_hypothesis = generalize(specific_hypothesis, instance)
            for idx, val in enumerate(specific_hypothesis):
                if val == '?':
                    general_hypothesis[idx] = '?'
                elif val != instance[idx]:
                    general_hypothesis[idx] = specific_hypothesis[idx]
        else:  # Negative instance
            new_general = []
            for gen, spec in zip(general_hypothesis, specific_hypothesis):
                if gen != spec:
                    new_general.append(gen)
            general_hypothesis = new_general[:]
        
        print("\nSpecific Hypothesis:", specific_hypothesis)
        print("General Hypothesis:", general_hypothesis)

# Read data from a CSV file
def read_csv(filename):
    data = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    return data

# Main function
def main():
    # Load training data from CSV file
    filename = r'C:\Users\Admin\Downloads\CarPrice.csv'
    training_data = read_csv(filename)

    # Display training data
    print("Training Data:")
    for row in training_data:
        print(row)

    # Apply Candidate-Elimination algorithm
    candidate_elimination(training_data)

if __name__ == "__main__":
    main()
