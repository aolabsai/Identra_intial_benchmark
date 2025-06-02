import kagglehub
import pandas as pd
import ao_core as ao
import numpy as np

Number_trials = 1000

num_train = Number_trials*0.8
num_test = Number_trials*0.2

Arch = ao.Arch(arch_i=[20,28], arch_z=[10])

# This seems like the best dataset to use since it is pretty very balenced , 1:1 

# Download latest version
path = kagglehub.dataset_download("nelgiriyewithana/credit-card-fraud-detection-dataset-2023")

# Load the dataset
df = pd.read_csv(path+ "/creditcard_2023.csv")
# Need some way to reduce the dimensaonality of the dataset 

# Vs 0-28 are anonymized features such as time, location, and other transaction details. I am going to be refering to v1 to v28 as feature embeddings.

"""

    index      id        V1        V2        V3        V4        V5        V6        V7        V8  ...       V21       V22       V23       V24       V25       V26       V27       V28    Amount  Class
0  268028  268028  1.738680 -0.539571  0.343099 -0.478460  0.007520 -0.014745  0.286824 -0.171451  ... -0.154207 -0.256802  0.338387 -0.135488 -0.670516 -2.436082 -0.171585 -0.168325   6581.16      0
1  303113  303113 -1.261394 -2.382550 -0.710428 -0.101410 -0.612158 -0.351779 -0.079762  0.151195  ...  0.077080 -0.046480 -2.568478 -0.637696 -0.412118 -0.766365 -0.126987 -2.419861  18331.09      1
2  460564  460564  0.933925 -0.303537  0.939915  0.301283  0.258574  0.767255  0.393672 -0.106449  ... -0.168061 -0.311770  0.040239  0.034921  0.489564 -0.237276 -0.218340 -0.082958  13572.00      1
3  267441  267441  0.205243 -0.073504  0.172032 -0.094131  0.326669 -0.171375  0.621503 -0.141087  ...  0.073212  1.307278 -0.027531  0.082045 -1.574954  0.367257  0.004532  0.415400  15238.23      0
4  143691  143691 -0.252363 -0.348904  1.333573 -0.709723  0.301442  0.239007  0.390342  0.005781  ... -0.061605  0.006577 -0.303214  0.097200  0.218744  0.659125 -0.319743 -0.261463   5581.06      0
"""

# shuffle the dataset

df = df.sample(frac=1).reset_index()

# print(df.head())


def convertAmountToBinary(amount):
    normalized_amount = (amount - df["Amount"].min()) / (df["Amount"].max() - df["Amount"].min())
    int_amount = int(normalized_amount * (2**20 - 1))  # Scale to fit in 20 bits
    binary_amount = format(int(int_amount), "020b")
    return binary_amount

def convertFeatureEmbeddingToBinary(embedding, threshold=0):
    """Converts a float32 embedding to a binary embedding."""
    binary_embedding = np.where(embedding > threshold, 1, 0)
    return binary_embedding

def run_ao_becnhmark():

    Agent = ao.Agent(Arch=Arch)

    correct = 0

    training_df = df.sample(frac=0.8, random_state=42)  # 80% for training
    test_df = df.drop(training_df.index)  # Remaining 20% for testing


    train_inputs = []
    train_outputs = []
    for row in training_df[0:int(num_train)].iterrows():
        amount  = row[1]["Amount"]
        class_type = row[1]["Class"]
        feature_embedding  = row[1][2:30].values  # V1 to V28
        



        binary_embedding = convertFeatureEmbeddingToBinary(feature_embedding) # This uses a guassian random projection to convert the feature embedding to a binary embedding
        binary_amount = convertAmountToBinary(amount)
        binary_amount_list = []
        
        for bit in binary_amount:
            binary_amount_list.append(int(bit))

        binary_amount = binary_amount_list  # Convert string to list of integers

        input_data = np.append(binary_embedding, binary_amount)  # Combine the binary embedding with the binary amount

        train_inputs.append(input_data)
        train_outputs.append([class_type]*10)




        if row[0] % 100 == 0:
            print("Trial: ", row[0])

    Agent.next_state_batch(INPUT=train_inputs, LABEL=train_outputs, unsequenced=True)


    for row in test_df[0:int(num_test)].iterrows():
        amount  = row[1]["Amount"]
        class_type = row[1]["Class"]
        feature_embedding  = row[1][2:30].values  # V1 to V28, 

        binary_embedding = convertFeatureEmbeddingToBinary(feature_embedding) 
        binary_amount = convertAmountToBinary(amount)
        binary_amount_list = []
        
        for bit in binary_amount:
            binary_amount_list.append(int(bit))

        binary_amount = binary_amount_list  # Convert string to list of integers

        input_data = np.append(binary_embedding, binary_amount)  # Combine the binary embedding with the binary amount

        for i in range(5): # For convergence, we will run the agent 5 times
            raw_response = Agent.next_state(INPUT=input_data)

        Agent.reset_state()


        response = 0
        if sum(raw_response) > 5:
            response = 1
        else:
            response = 0  

        # Uncomment if we want to train in real-time

        # if class_type == 1:
        #     Agent.next_state(INPUT=input_data, LABEL=[1,1,1,1,1,1,1,1,1,1])
        # else:
        #     Agent.next_state(INPUT=input_data, LABEL=[0,0,0,0,0,0,0,0,0,0])

        if response == class_type:
            correct += 1

        if row[0] % 100 == 0:
            print("Trial: ", row[0], " Correct: ", correct)

    return correct


if __name__ == "__main__":
    num_tests = [10, 100]
    accuracy_list = []
    for num_test in num_tests:
        print(f"Running benchmark with {num_test} tests...")
        amount_correct = run_ao_becnhmark()
        accuracy = amount_correct / num_test
        accuracy_list.append(accuracy)
        print("Accuracy: ", accuracy)

        
print("Tests: ", num_tests)
print("Final Accuracy List: ", accuracy_list)


