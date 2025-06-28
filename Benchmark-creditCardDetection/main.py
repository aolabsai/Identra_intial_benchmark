import kagglehub
import pandas as pd
import ao_core as ao
from ao_embeddings import binaryEmbeddings as be
from config import OPENAI_KEY
import numpy as np
from datetime import datetime

type_of_conversion = "q"    # or "gaussian" for Gaussian random projection. Threshold seems to work better 


# This seems like the best dataset to use since it is very balenced , 1:1 (https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)

# Download latest version
path = kagglehub.dataset_download("nelgiriyewithana/credit-card-fraud-detection-dataset-2023")

# Load the dataset
df = pd.read_csv(path+ "/creditcard_2023.csv")
n_bits = 6 # number of bits per feature
n_levels = 2 ** n_bits # the number of possible levels for each feature

if type_of_conversion not in ["threshold", "gaussian", "q"]:
    raise ValueError("type_of_conversion must be either 'threshold' or 'gaussian'")
elif type_of_conversion == "gaussian":
    em = be.binaryEmbeddings(openai_api_key=OPENAI_KEY, cacheName="embeddingCache.json", numberBinaryDigits=28)
elif type_of_conversion == "q":
    # bins_per_feature will be a list of 28 arrays, each of length (n_levels+1)
    bins_per_feature = []
    for feat_idx in range(28):
        col = df[f"V{feat_idx+1}"].values
        # define bin‐edges so that you get n_levels “buckets” between min and max
        edges = np.linspace(col.min(), col.max(), num=n_levels+1) # calc range
        bins_per_feature.append(edges)

    def quantize_vector(feat_vec, bins_list, n_bits=3):
        """feat_vec shape=(28,), bins_list is length‐28 list of edges arrays."""
        code_bits = []
        for v, edges in zip(feat_vec, bins_list):
            # np.digitize returns an integer in [1..len(edges)], so subtract 1
            level = np.digitize(v, edges) - 1  
            # clip just in case
            level = max(0, min(level, n_levels-1))
            # turn into n_bits binary string
            code_bits.extend(int(b) for b in format(level, f"0{n_bits}b"))
        return code_bits
np.random.seed(42)  # For reproducibility

# Vs 0-28 are anonymized features such as time, location, and other transaction details. I am going to be refering to v1 to v28 as feature embeddings.

"""

    id        V1        V2        V3        V4        V5        V6        V7        V8        V9       V10  ...       V20       V21       V22       V23       V24       V25       V26       V27       V28    Amount  Class
0   0 -0.260648 -0.469648  2.496266 -0.083724  0.129681  0.732898  0.519014 -0.130006  0.727159  0.637735  ...  0.091202 -0.110552  0.217606 -0.134794  0.165959  0.126280 -0.434824 -0.081230 -0.151045  17982.10      0
1   1  0.985100 -0.356045  0.558056 -0.429654  0.277140  0.428605  0.406466 -0.133118  0.347452  0.529808  ... -0.233984 -0.194936 -0.605761  0.079469 -0.577395  0.190090  0.296503 -0.248052 -0.064512   6531.37      0
2   2 -0.260272 -0.949385  1.728538 -0.457986  0.074062  1.419481  0.743511 -0.095576 -0.261297  0.690708  ...  0.361652 -0.005020  0.702906  0.945045 -1.154666 -0.605564 -0.312895 -0.300258 -0.244718   2513.54      0
3   3 -0.152152 -0.508959  1.746840 -1.090178  0.249486  1.143312  0.518269 -0.065130 -0.205698  0.575231  ... -0.378223 -0.146927 -0.038212 -0.214048 -1.893131  1.003963 -0.515950 -0.165316  0.048424   5384.44      0
4   4 -0.206820 -0.165280  1.527053 -0.448293  0.106125  0.530549  0.658849 -0.212660  1.049921  0.968046  ...  0.247237 -0.106984  0.729727 -0.161666  0.312561 -0.414116  1.071126  0.023712  0.419117  14278.97      0
"""

# shuffle the dataset


df = df.sample(frac=1).reset_index()

correct = 0

def float_to_binary(embedding, threshold=0):  # The most basic conversion function. if input is greater than threshold, it will be 1, else 0
  """Converts a float32 embedding to a binary embedding."""
  binary_embedding = np.where(embedding > threshold, 1, 0)
  return binary_embedding


def convertAmountToBinary(amount):
    normalized_amount = (amount - df["Amount"].min()) / (df["Amount"].max() - df["Amount"].min()) # Normalize the amount using the formula: (value - min) / (max - min)
    int_amount = int(normalized_amount * (2**20 - 1))  # Scale to fit in 20 bits
    binary_amount = format(int(int_amount), "020b")
    return binary_amount

def convertFeatureEmbeddingToBinary(feature_embedding):
    """Converts a feature embedding to a binary embedding"""
    if type_of_conversion == "threshold":
        binary_code = float_to_binary(feature_embedding)#
    elif type_of_conversion == "gaussian":
        # Gaussian random projection to binary
        # mean = np.mean(feature_embedding)
        # std_dev = np.std(feature_embedding)
        # gaussian_projection = (feature_embedding - mean) / std_dev
        # binary_code = float_to_binary(gaussian_projection, threshold=0)

        binary_code = em.embeddingToBinary(feature_embedding)
    elif type_of_conversion == "q":
        binary_code = quantize_vector(feature_embedding, bins_per_feature, n_bits)

    return binary_code

def runAOModel(Number_trials):
    correct = 0

    num_train = Number_trials*0.8
    num_test = Number_trials*0.2

    if type_of_conversion == "q":
        Arch = ao.Arch(arch_i=[(28*n_bits),20], arch_z=[10])
    else:
        Arch = ao.Arch(arch_i=[28,20], arch_z=[10])
    Agent = ao.Agent(Arch=Arch)

    training_df = df.sample(frac=0.8, random_state=42)  # 80% for training
    test_df = df.drop(training_df.index)  # Remaining 20% for testing


    train_inputs = []
    train_outputs = []
    for i, row in enumerate(training_df[0:int(num_train)].iterrows()):

        amount  = row[1]["Amount"]
        class_type = row[1]["Class"]
        feature_embedding  = row[1][1:29].values  # V1 to V28
        



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
            print("Trial: ", i)

    Agent.next_state_batch(INPUT=train_inputs, LABEL=train_outputs, unsequenced=True)


    for row in test_df[0:int(num_test)].iterrows():
        amount  = row[1]["Amount"]
        class_type = row[1]["Class"]
        feature_embedding  = row[1][1:29].values  # V1 to V28

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

    print("amount correct: ", correct/num_test)
    return correct/num_test

def runTrials(Number_trials):
    ao_acc = runAOModel(Number_trials)
    return ao_acc


if __name__ == "__main__":
    trials_array= [10, 100, 1000]
    acc_array = []
    times_array = []
    for trials in trials_array:
        now = datetime.now()
        print("Running trials: ", trials)
        acc = runTrials(trials)
        acc_array.append(acc)
        print("Accuracy for trials ", trials, ": ", acc)
        later = datetime.now()
        time_diff = later - now
        times_array.append(time_diff.total_seconds())

    results = []
    for i in range(len(trials_array)):
        results.append((trials_array[i], acc_array[i], times_array[i]))
    print("Results: ", results)
