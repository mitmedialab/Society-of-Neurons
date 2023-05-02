import tensorflow as tf
import numpy as np

# Define the path to the saved model
model_path = 'path/to/your/saved/model'

# Load the saved model
loaded_model = tf.keras.models.load_model(model_path)

# Get the number of layers in the model
num_layers = len(loaded_model.layers)

# Create an empty adjacency matrix with dimensions equal to the number of layers
adj_matrix = np.zeros((num_layers, num_layers))

# Iterate through each layer and its inbound layers, and mark the corresponding entries in the adjacency matrix as 1
for i, layer in enumerate(loaded_model.layers):
    for inbound_node in layer._inbound_nodes:
        for inbound_layer in inbound_node.inbound_layers:
            j = loaded_model.layers.index(inbound_layer)
            adj_matrix[i][j] = 1

# Print the adjacency matrix
print("Adjacency Matrix:")
print(adj_matrix)
