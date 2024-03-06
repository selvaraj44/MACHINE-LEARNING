import csv

# Define the find-s algorithm
def find_s_algorithm(training_data):
    # Initialize hypothesis with the first training instance
    hypothesis = training_data[0][:-1]  # Take the attributes of the first instance
    for instance in training_data:
        if instance[-1] == 'Yes':  # Check if the instance is positive
            for i in range(len(hypothesis)):
                if instance[i] != hypothesis[i]:
                    hypothesis[i] = '?'  # Update hypothesis to include the attribute value
    return hypothesis

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

    # Apply FIND-S algorithm
    hypothesis = find_s_algorithm(training_data)

    # Display the hypothesis
    print("\nHypothesis (Most Specific):", hypothesis)

if __name__ == "__main__":
    main()
