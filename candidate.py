import csv

# Read CSV file
with open("job_data.csv") as file:
    data = list(csv.reader(file))

# Separate header and training data
header = data[0]
training_data = data[1:]

print("Header:", header)
print("\nTraining Data:")
for row in training_data:
    print(row)                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
# Initialize Specific Boundary (S)
S = training_data[0][:-1]

# Initialize General Boundary (G)
G = [["?" for i in range(len(S))] for i in range(len(S))]

# Candidate Elimination Algorithm
for i, example in enumerate(training_data):

    if example[-1] == "Yes":  # Positive Example
        for j in range(len(S)):
            if example[j] != S[j]:
                S[j] = "?"
                G[j][j] = "?"

    elif example[-1] == "No":  # Negative Example
        for j in range(len(S)):
            if example[j] != S[j]:
                G[j][j] = S[j]
            else:
                G[j][j] = "?"

    print("\nStep", i + 1)
    print("S =", S)
    print("G =", G)

print("\nFinal Output")
print("Specific Boundary (S):", S)
print("General Boundary (G):", G)
