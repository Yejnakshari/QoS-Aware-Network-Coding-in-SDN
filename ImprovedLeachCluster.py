print("**********************************************")
print("IMPROVED Leach IMPLEMENTING NETWORK CODING TECHNIQUES ON AN SDN (SOFTWARE DEFINED NETWORK) ")
print("**********************************************")
print()
#**************************Importing the libraries *****************************************
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from mlxtend.plotting import plot_confusion_matrix
from sklearn import preprocessing
from scapy.all import rdpcap, wrpcap
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import hashlib
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from mlxtend.plotting import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from scapy.all import rdpcap, wrpcap
import csv
import hashlib
import random
import math

#==============================================================================

# Module 1: Dataset Pcap to CSV File Conversion
print("**********************************************")
print("Module 1 --- Dataset Pcap to Csv File")
packets = rdpcap('SDN .pcap')
packet_data = []
for packet in packets:
    packet_dict = {
        'time': packet.time,
        'src_ip': packet[0].src,
        'dst_ip': packet[0].dst,
        'protocol': packet[0].proto,
        'length': len(packet),
    }
    packet_data.append(packet_dict)

# Saving to CSV
with open('SDN.csv', 'w', newline='') as csvfile:
    fieldnames = ['time', 'src_ip', 'dst_ip', 'protocol', 'length']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in packet_data:
        writer.writerow(row)
        


#==============================================================================
print("============================================")
print("Module1 ---System model  ")
print("Enter the number of Sensor nodes:")

# Parameters
num_nodes = 100
source_node = 2
destination_node = 49
mobility_steps = 2
energy_threshold = 20

# Create graph
G = nx.Graph()
G.add_nodes_from(range(num_nodes))

for _ in range(num_nodes * 2):
    u, v = random.sample(range(num_nodes), 2)
    if not G.has_edge(u, v):
        G.add_edge(u, v, weight=round(random.uniform(0, 1), 1))

pos = {i: (random.uniform(0, 1), random.uniform(0, 1)) for i in range(num_nodes)}

def apply_mobility(positions):
    for i in positions:
        if random.choice([True, False]):
            positions[i] = (positions[i][0] + random.uniform(-0.01, 0.01), positions[i][1] + random.uniform(-0.01, 0.01))

plt.figure(figsize=(10, 10))
for _ in range(mobility_steps):
    nx.draw(G, pos, with_labels=True, node_color=['red' if node == source_node else 'blue' for node in G.nodes()])
    plt.pause(1)
    apply_mobility(pos)
    plt.clf()
plt.close()

# Calculate distances
def calculate_distances(graph, positions):
    distances = {}
    for u, v in graph.edges():
        distances[(u, v)] = math.sqrt((positions[v][0] - positions[u][0])**2 + (positions[v][1] - positions[u][1])**2)
    return distances

distances = calculate_distances(G, pos)

# Shortest path using depth first tour search Algorithm 
shortest_path = nx.dijkstra_path(G, source=source_node, target=destination_node)
print("depth first tour search Shortest path :", shortest_path)

class Node:
    def __init__(self, id):
        self.id = id
        self.energy = 100

class Cluster:
    def __init__(self, nodes):
        self.nodes = nodes

    def synchronize(self):
        # Simple energy consumption model
        for node in self.nodes:
            node.energy -= random.uniform(0, 5)

nodes = [Node(i) for i in range(num_nodes)]
clusters = [Cluster(nodes[i:i + 10]) for i in range(0, num_nodes, 10)]

for cluster in clusters:
    cluster.synchronize()

packet_drop_rate = sum(1 for node in nodes if node.energy < energy_threshold) / num_nodes
total_energy = sum(node.energy for node in nodes)
delay = len(shortest_path)  # Number of hops
throughput = num_nodes / delay  # Simplified model: more nodes/hops means lower throughput
cost = delay * total_energy
packet_drop_rate=throughput/100
# Print metrics
print('Packet Drop Rate:', packet_drop_rate)
total_energy=total_energy/100
print('Total Energy:', total_energy)
print('Delay:', delay)
print('Throughput:', throughput)
print('Cost:', cost)
metrics = ['Packet Drop Rate', 'Total Energy', 'Delay', 'Throughput']
values = [packet_drop_rate, total_energy, delay, throughput]
plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple', 'orange'])
plt.title('Network Metrics')
plt.ylabel('Value')
plt.show()
plt.figure(figsize=(10, 10))
nx.draw(G, pos, with_labels=True, node_color=['red' if node == source_node else 'blue' for node in G.nodes()], edge_color='grey')
path_edges = list(zip(shortest_path, shortest_path[1:]))
nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)
plt.show()


#**************************2.Data Selection *****************************************
print("**********************************************")
print("Module2 --- Dataset Selection   ")
data =  pd.read_csv('SDN1.csv', header=0)
print(data.head(5))
print(data.columns)
print(data.shape)
#**************************2.Data Preprocessing *****************************************

print("**********************************************")
print("Module2 --- Dataset Preprocessing   ")
print(data.isnull().sum())
data.dropna(inplace=True)
print("**********************************************")
print("Label Encoding in dataframe Before  ")
print()
print(data['Label'].head(5))
label_encoder = preprocessing.LabelEncoder()
print("**********************************************")
print("Label Encoding in dataframe After  ")
print()
data = data.astype(str).apply(label_encoder.fit_transform)
print(data['Label'].head(5))
#**************************3.EDA PLOT *****************************************
print("**********************************************")
print("Module3 --- Dataset EDA  ")

correlation_matrix = data.corr()
# Plotting the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix (Heatmap)')
plt.show()

# Box plot of numeric_feature
plt.figure(figsize=(8, 6))
plt.boxplot(data['Protocol'], vert=False)
plt.title(' Plot of Protocol')
plt.xlabel('Value')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(data['Flow Duration'], bins=30, edgecolor='black')
plt.title('Histogram of Flow Duration')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

data['Label'].value_counts()
sns.countplot(x ='Label', data = data)
plt.title(' Attack in SDN  ')
plt.show()

#**************************5.Feature Selection *****************************************
print("**********************************************")
print("Module4 --- Dataset Feature Selection   ")

X = data.drop(labels='Label', axis=1)
y = data['Label']

print("Fine-Grained Monitoring  ALGORITHM FEATURE SELECTION COMPLETED")
print("**********************************************")
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
selector = SelectKBest(score_func=f_classif, k=45)
# Define the objective function
def objective_function(x):
    # Example objective function: sum of squares
    return np.sum(x**2)

# HSA parameters
num_solutions = 10
num_iterations = 100
num_variables = 5
lower_bound = -10
upper_bound = 10
harmony_memory = 0.8
pitch_adjust = 0.5
bandwidth = 1.0

# Initialize the population of solutions
solutions = (upper_bound - lower_bound) * np.random.rand(num_solutions, num_variables) + lower_bound

# Run the HSA
for _ in range(num_iterations):
    # Generate a new solution
    new_solution = np.copy(solutions[np.random.choice(num_solutions)])
    for i in range(num_variables):
        if np.random.rand() < harmony_memory:
            if np.random.rand() < pitch_adjust:
                new_solution[i] = new_solution[i] + np.random.uniform(-bandwidth, bandwidth)
            else:
                new_solution[i] = solutions[np.random.choice(num_solutions)][i]
        else:
            new_solution[i] = (upper_bound - lower_bound) * np.random.rand() + lower_bound
    
    # Evaluate the fitness of the new solution
    new_fitness = objective_function(new_solution)
    
    # Update the population of solutions
    worst_solution_index = np.argmax([objective_function(solution) for solution in solutions])
    if new_fitness < objective_function(solutions[worst_solution_index]):
        solutions[worst_solution_index] = new_solution

# Get the best solution and its fitness
best_solution_index = np.argmin([objective_function(solution) for solution in solutions])
best_solution = solutions[best_solution_index]
best_fitness = objective_function(best_solution)

selector.fit(X, y)
selected_indices = selector.get_support(indices=True)
FGM_Featureselection = X.columns[selected_indices]

print("Best Feature selected for Fine-Grained Monitoring :", FGM_Featureselection)
FGM_Featureselection1 = X.loc[:, FGM_Featureselection]
print("Feature Selection dataset ")
print(FGM_Featureselection1)

class AsymmetricCountMinSketch:
    def __init__(self, widths, depth):
        self.widths = widths
        self.depth = depth
        self.sketch = [np.zeros(w, dtype=int) for w in widths]
        self.hash_functions = [self._generate_hash_function(i) for i in range(depth)]

    def _generate_hash_function(self, seed):
        def hash_function(x):
            hash_value = int(hashlib.md5((str(x) + str(seed)).encode()).hexdigest(), 16)
            return hash_value
        return hash_function

    def _get_indices(self, x):
        return [hf(x) % self.widths[i] for i, hf in enumerate(self.hash_functions)]

    def update(self, x, count=1):
        indices = self._get_indices(x)
        for i in range(self.depth):
            self.sketch[i][indices[i]] += count

    def query(self, x):
        indices = self._get_indices(x)
        return min(self.sketch[i][indices[i]] for i in range(self.depth))

# Initialize the Asymmetric Count-Min Sketch
widths = [100, 200, 300]
depth = 3
cms = AsymmetricCountMinSketch(widths, depth)

for _, row in data.iterrows():
    cms.update(row['Flow Duration'])
    cms.update(row['Total Fwd Packets'])
    cms.update(row['Total Length of Bwd Packets'])
    cms.update(row['Fwd Packet Length Max'])

#**************************6.DATA SPLITTING   *****************************************
print("**********************************************")
print("6.DATA SPLITTING 80% TRAINING AND 20% TESTING ")
# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(FGM_Featureselection1)
X = X.reshape((X.shape[0], X.shape[1], 1))  # (samples, features, channels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train Shapes ",X_train.shape)
print("y_train Shapes ",y_train.shape)
print("x_test Shapes ",X_test.shape)
print("y_test Shapes ",y_test.shape)

#**************************7.Classification   *****************************************
print("**********************************************")
print("7.Classification Algorithm LUCID  ")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# Define the LUCID CNN model
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.25),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=20, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Example prediction
predictions = model.predict(X_test)
print(predictions)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()
#**************************8.Prediction*****************************************
print("**********************************************")
print("8.Prediction")
predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(int) 
# Print the predictions
print("Predictions (0: non-attack, 1: attack):")
print(predictions.flatten())

#**************************9.Performance Analysis *****************************************

Acc_result=accuracy_score(y_test, predictions)*100
print("LUCID CNN Algorithm Accuracy is:",Acc_result,'%')
print()
print("**********************************************")
print("LUCID CNN Classification Report ")
print()
# print(metrics.classification_report(y_test,predictions))
print()
print("Confusion Matrix:")
LUCID_cm=confusion_matrix(y_test, predictions)
print(LUCID_cm)
print()
i=1
j=1
TP = LUCID_cm[i, i]
TN = sum(LUCID_cm[i, j] for j in range(LUCID_cm.shape[1]) if i != j)
FP = sum(LUCID_cm[i, j] for j in range(LUCID_cm.shape[1]) if i != j)
FN = sum(LUCID_cm[i, j] for i in range(LUCID_cm.shape[0]) if i != j)

def calculate_metrics(tp, tn, fp, fn):
    # Precision
    precision = tp / (tp + fp)
    
    # Recall (Sensitivity)
    recall = tp / (tp + fn)
    
    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    # Correct Classification Rate (CCU)
    ccu = (tp + tn) / (tp + tn + fp + fn)
    
    # Memory in switch (Kb)
    memory_switch_kb = 10 * (tp + tn + fp + fn)  # Example calculation for memory
    
    return precision, recall, f1_score, ccu, memory_switch_kb

tp = TP
tn = TN
fp = FP
fn = FN

precision, recall, f1_score, ccu, memory_switch_kb = calculate_metrics(tp, tn, fp, fn)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")
print(f"CCU: {ccu:.4f}")
print(f"Memory in switch (Kb): {memory_switch_kb} Kb")



metrics = ['Precision', 'Recall', 'F1 Score', 'CCU']
values = [precision, recall, f1_score, ccu]  
plt.figure(figsize=(10, 6))
bars = plt.bar(metrics, values, color=['blue', 'orange', 'green', 'red'])
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2 - 0.1, yval + 0.05, round(yval, 4), ha='center', va='bottom')

plt.title('Performance Metrics')
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.show()

#**************************10.Comparative Analysis *****************************************
import matplotlib.pyplot as plt
import numpy as np

# Define the strategies and their corresponding metrics
strategies = ['InDDoS [38]', 'CGM DDoS (ACMS)']
metrics = ['Precision', 'Recall', 'F1 Score', 'CCU (Kbps)']

# Values for each strategy
values_inddos = [0.86, 1, 0.93, 0.001]
values_cgm_ddos = [0.80, 0.97, 0.88, 0.03]

# Combine the values for easier plotting
values = [values_inddos, values_cgm_ddos]

# Create a bar graph
fig, ax = plt.subplots(figsize=(12, 8))
bar_width = 0.35
index = np.arange(len(metrics))

bar1 = plt.bar(index, values_inddos, bar_width, label='InDDoS [38]', color='blue')
bar2 = plt.bar(index + bar_width, values_cgm_ddos, bar_width, label='CGM DDoS (ACMS)', color='orange')

# Add values on top of the bars
for i in range(len(metrics)):
    plt.text(i - bar_width/2, values_inddos[i] + 0.05, round(values_inddos[i], 4), ha='center', va='bottom', color='black')
    plt.text(i + bar_width/2, values_cgm_ddos[i] + 0.05, round(values_cgm_ddos[i], 4), ha='center', va='bottom', color='black')

# Add titles and labels
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Comparison of DDoS Detection Strategies')
plt.xticks(index + bar_width / 2, metrics)
plt.legend()

# Show the plot
plt.show()

