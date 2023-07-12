'''
Full Pipeline for the Society of Neurons Project to understand the process of LLMs

******************Data workflow******************

question_id is the unique identifier across the whole project

- Set of questions → csv file
    - question id
    - prompt
    - groundtruth answer
    - question class
- Activations → .pt files
    - question id
    - hidden_states
    - output
- Processed activations dataframe (for faster loading) → .csv
    - columns : q_ids
    - rows: neuron_ids (flattened)
- Cluster → list of tuples (.npy files)
    - [(layer_id, emb_id), ..]
- Analysis files → csv
    - question id
    - accuracy without knockout
    - accuracy with knockout

******************Function workflow******************

- load_llm: load an llm for evaluation
    - input → model_paths
    - output → model instance
- generate_outputs_batch: takes a set of questions and generates llm outputs, can do knockout
    - input → set of questions, llm, optional cluster npy file, activation dir
    - output → activations, save to activation_dir
- process_activations: load activation dir using aggregation strategy (first/avg/last) and save the df
    - input → activations_dir, set_of_questions
    - output → Processed activations dataframe
- cluster_activations: load activations df and do clustering and save them, append _kmeans or _pca for the method, optionally calculate r2 score to rank clusters
    - input → Processed activations dataframe, clustering_kwargs
    - output → cluster npy files saved
'''

import os 
import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from sklearn.cluster import KMeans

opj = os.path.join

def load_llm(base_model_path, finetune_model_path):

    tokenizer = LlamaTokenizer.from_pretrained(base_model_path)
    print("Loaded tokenizer")
    
    model = LlamaForCausalLM.from_pretrained(
        base_model_path,
        load_in_8bit=True,
        device_map="auto",
    )
    print("Loaded base model")

    model = PeftModel.from_pretrained(model, finetune_model_path)
    print("Loaded full model")

    return tokenizer, model


def generate_output(input_prompt, tokenizer, model, knockout_neurons):
    """
    Takes the instruction, puts it in the instruction finetuning template and returns the model generated output, along with the hidden states
    """

    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{input_prompt}

### Response:"""

    print("Prompt:", input_prompt)

    generation_config = GenerationConfig(
            temperature=0,
            top_p=1,
            num_beams=1, # beam search
            )

    if knockout_neurons is not None:
        print("Knocking out neurons")

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=256,
        output_hidden_states=True,
        knockout_neurons = knockout_neurons,
        ns_value=1.0
    )

    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        print("Response:", output.split("### Response:")[1].strip())

    return generation_output


def generate_outputs_batch(csv_files, tokenizer, model, knockout_cluster=None, activations_dir=None):
    """
    Generates outputs for a set of questions using a language model (llm),
    and saves the outputs in a specified directory.
    """

    if type(csv_files) == str:
        csv_files = [csv_files]

    set_of_questions = []
    for c in csv_files:
        data = pd.read_csv(c)
        selected_columns = data[['Question_ID', 'Question', 'Question_Type']]
        set_of_questions.extend([tuple(x) for x in selected_columns.to_numpy()][:10])

    print(f"Loaded {len(set_of_questions)} set of questions")

    if knockout_cluster:
        knockout_neurons = np.load(knockout_cluster)
    else:
        knockout_neurons = None

    if activations_dir:
        if not os.path.exists(activations_dir):
            os.makedirs(activations_dir)

    # Loop over all questions in the set
    for q_id, q, _ in set_of_questions:
        print(f"Processing {q_id}")

        # Generate output using the language model
        generation_output = generate_output(q, tokenizer, model, knockout_neurons)

        if activations_dir:
            output_to_save = generation_output
            output_to_save['prompt'] = q
            output_to_save['question_id'] = q_id

            for s in generation_output.sequences:
                output_tokens = tokenizer.decode(s)

            output_to_save['output'] = output_tokens

            # del output_to_save.sequences

            save_path = opj(activations_dir, f"{q_id}.pt")
            torch.save(output_to_save, save_path)
    print("Processed all questions")
    

def process_activations(csv_files, activation_dir, aggr_strategy='first'):
    """
    load activation dir using aggregation strategy (first/avg/last) and save the df in format of q_idxneuron_id
    """
    
    if type(csv_files) == str:
        csv_files = [csv_files]

    set_of_questions = []
    for c in csv_files:
        data = pd.read_csv(c)
        selected_columns = data[['Question_ID', 'Question', 'Question_Type']]
        set_of_questions.extend([tuple(x) for x in selected_columns.to_numpy()][:10])
    

    print(f"Loaded {len(set_of_questions)} set of questions")

    activations_df = pd.DataFrame()
    
    for q_id, _,  _ in tqdm(set_of_questions):
        activation_file = opj(activation_dir, f"{q_id}.pt")
        output_data = torch.load(activation_file, map_location=torch.device('cpu'))
        hidden_states = output_data['hidden_states']

        if aggr_strategy == "first":
            activations = []
            token_hidden_states = hidden_states[3]
            for layers in token_hidden_states:
                    for token_activations in layers[0]: # single beam search
                        token_activations_np = token_activations.numpy()
                        activations.extend(token_activations_np)

        elif aggr_strategy == "avg":
            activations = []
            token_hidden_states = hidden_states[3:]
            avg_activations = None
            for t in token_hidden_states:
                for layers in t:
                    for token_activations in layers[0]: # single beam search
                        token_activations_np = token_activations.numpy()
                        activations.extend(token_activations_np)

                if avg_activations is not None:
                    avg_activations += np.array(activations)
                else:
                    avg_activations = np.array(activations)

            activations = np.mean(avg_activations, axis=0)

        activations_df[q_id] = activations

    activations_df_save_path = opj(activation_dir, f'activations_{aggr_strategy}.csv')
    activations_df.to_csv(activations_df_save_path)
    print(f"Saved activations df to {activations_df_save_path}")


def cluster_activations_kmeans(activations_df_file, clusters_dir, cluster_kwargs={}, hidden_size=5120):
    """
    load activations df and do clustering, analyze and save them, append _kmeans or _pca for the method, 
    optionally calculate r2 score to rank clusters
    """

    activations_df = pd.read_csv(activations_df_file)

    if 'n_clusters' in cluster_kwargs.keys():
        n_clusters = cluster_kwargs['n_clusters']
    else:
        n_clusters = 14

    kmeans = KMeans(**cluster_kwargs, random_state=42)
    kmeans.fit(activations_df)
    pred = kmeans.predict(activations_df)

    for c_id in range(n_clusters):
        cluster = []
        for e, p in enumerate(pred):
            if p == c_id:
                cluster.append((e//hidden_size, e%hidden_size))

        cluster_save_path = opj(clusters_dir, f"{c_id}")
        np.save(cluster_save_path, cluster)
        print(f"Saved cluster of {len(cluster)} neurons with id {c_id} to {cluster_save_path}")


def cluster_activations_pca(activations_df_file, clusters_dir, cluster_kwargs={}, hidden_size=5120):
    """
    load activations df and do clustering, analyze and save them, append _kmeans or _pca for the method, 
    optionally calculate r2 score to rank clusters
    """
    pass


def visualize_activations(q_id, activation_dir):
    pass
