import os
opj = os.path.join

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

import utils

print("Started pipeline")

# load llm
model_path = '/home/gridsan/wzulfikar/models/llama-13b-hf'
lora_model_path = '/home/gridsan/wzulfikar/models/alpaca-lora-13b' #chansung

tokenizer, model = utils.load_llm(model_path, lora_model_path)

# generate baseline outputs
dataset_dir = '/home/gridsan/wzulfikar/activations/cot/'

cot_csv_file = opj(dataset_dir, 'prompt_chain.csv')
no_cot_csv_file = opj(dataset_dir, 'prompt_no_chain.csv')
raw_activations_dir = opj(dataset_dir, 'raw')

utils.generate_outputs_batch([cot_csv_file, no_cot_csv_file], tokenizer, model, activations_dir=raw_activations_dir)


# process activations for clustering
aggr_strategy = 'first'
utils.process_activations([cot_csv_file, no_cot_csv_file], raw_activations_dir, aggr_strategy=aggr_strategy)

# cluster using kmeans
activations_df_file = opj(raw_activations_dir, f'activations_{aggr_strategy}.csv')
clusters_dir = opj(dataset_dir, 'clusters')
n_clusters = 16

utils.cluster_activations_kmeans(activations_df_file, clusters_dir, cluster_kwargs={'n_clusters': n_clusters})

# try dataset on knockout of each cluster
for c in range(n_clusters):
    knockout_cluster = opj(clusters_dir, f'{c}.npy')
    knockout_activation_dir = opj(clusters_dir, f'cluster_{c}')
    generate_outputs_batch([cot_csv_file, no_cot_csv_file], tokenizer, model, knockout_cluster=knockout_cluster, activations_dir=knockout_activation_dir)