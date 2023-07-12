import os
opj = os.path.join

from utils import load_llm, generate_outputs_batch
from utils import process_activations, cluster_activations_kmeans

# load llm
model_path = './models/llama-13b-hf'
lora_model_path = './models/alpaca-lora-13b' #chansung

tokenizer, model = load_llm(model_path, lora_model_path)

# generate baseline outputs
dataset_dir = './activations/cot/'

cot_csv_file = opj(dataset_dir, 'prompt_chain.csv')
no_cot_csv_file = opj(dataset_dir, 'prompt_no_chain.csv')
raw_activations_dir = opj(dataset_dir, 'raw')

generate_outputs_batch([cot_csv_file, no_cot_csv_file], tokenizer, model, activations_dir=raw_activations_dir)


# process activations for clustering
aggr_strategy = 'avg'
process_activations([cot_csv_file, no_cot_csv_file], raw_activations_dir, aggr_strategy=aggr_strategy)

# cluster using kmeans
activations_df_file = opj(raw_activations_dir, f'activations_{aggr_strategy}.csv')
clusters_dir = f'./clusters/cot'
n_clusters = 16

cluster_activations_kmeans(activations_df_file, clusters_dir, cluster_kwargs={'n_clusters': n_clusters})

# try dataset on knockout of each cluster
for c in range(n_clusters):
    knockout_activation_dir = opj(dataset_dir, f'{c}.npy')
    generate_outputs_batch([cot_csv_file, no_cot_csv_file], tokenizer, model, activations_dir=knockout_activation_dir)
