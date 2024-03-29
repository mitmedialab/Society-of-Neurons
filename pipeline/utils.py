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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation
import matplotlib

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
        num_beams=1,  # beam search
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
        knockout_neurons=knockout_neurons,
        ns_value=0.0
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
        set_of_questions.extend([tuple(x) for x in selected_columns.to_numpy()][:50])

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
        set_of_questions.extend([tuple(x) for x in selected_columns.to_numpy()][:50])

    print(f"Loaded {len(set_of_questions)} set of questions")

    activations_df = pd.DataFrame()

    for q_id, _, _ in tqdm(set_of_questions):
        activation_file = opj(activation_dir, f"{q_id}.pt")
        output_data = torch.load(activation_file, map_location=torch.device('cpu'))
        hidden_states = output_data['hidden_states']

        if aggr_strategy == "first":
            activations = []
            token_hidden_states = hidden_states[3]
            for layers in token_hidden_states:
                for token_activations in layers[0]:  # single beam search
                    token_activations_np = token_activations.numpy()
                    activations.extend(token_activations_np)

        elif aggr_strategy == "avg":
            token_hidden_states = hidden_states[3:]
            avg_activations = None
            for t in token_hidden_states:
                activations = []
                for layers in t:
                    for token_activations in layers[0]:  # single beam search
                        token_activations_np = token_activations.numpy()
                        activations.extend(token_activations_np)

                if avg_activations is not None:
                    avg_activations += np.array(activations)
                else:
                    avg_activations = np.array(activations)

            activations = avg_activations / len(token_hidden_states)

        activations_df[q_id] = activations

    activations_df_save_path = opj(activation_dir, f'activations_{aggr_strategy}.csv')
    activations_df.to_csv(activations_df_save_path, index=False)
    print(f"Saved activations df to {activations_df_save_path}")


def cluster_activations_kmeans(activations_df_file, clusters_dir, cluster_kwargs={}, hidden_size=5120,
                               calculate_significance=False):
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
                cluster.append((e // hidden_size - 1, e % hidden_size))

        cluster_save_path = opj(clusters_dir, f"{c_id}")
        np.save(cluster_save_path, cluster)
        print(f"Saved cluster of {len(cluster)} neurons with id {c_id} to {cluster_save_path}.npy")

        if calculate_significance:

            cluster_indices = [i[0] * hidden_size + i[1] for i in cluster]
            if len(cluster_indices) == 0:
                continue

            cluster_activations = activations_df.iloc[cluster_indices].T

            task_labels = np.array([0] * (activations_df.shape[1] // 2) + [1] * (activations_df.shape[1] // 2))

            # convert cluster_activations to numpy array
            cluster_activations = cluster_activations.to_numpy()

            # do linear regression
            lr = LinearRegression()
            lr.fit(cluster_activations, task_labels)

            pred_r2 = lr.predict(cluster_activations)

            r2_score_value = r2_score(pred_r2, task_labels)

            print(f"R2 score of cluster {c_id} is {r2_score_value}")


def cluster_activations_pca(activations_df_file, clusters_dir, percentile=95, cluster_kwargs={}, hidden_size=5120,
                            calculate_significance=False):
    """
    load activations df and do PCA, analyze and save them, append _pca for the method, 
    optionally calculate r2 score to rank clusters
    """

    activations_df = pd.read_csv(activations_df_file)

    if 'n_components' in cluster_kwargs.keys():
        n_components = cluster_kwargs['n_components']
    else:
        n_components = 14

    pca = PCA(**cluster_kwargs)
    pca.fit(activations_df)
    components = pca.transform(activations_df)

    # Iterate through each principal component
    for c_id in range(n_components):
        component = []

        # threshold for determining if neuron significantly contributes to the component
        threshold = np.percentile(np.abs(components[:, c_id]), percentile)

        # Iterate over each element in the current principal component
        for e, p in enumerate(components[:, c_id]):
            if abs(p) >= threshold:
                component.append((e // hidden_size - 1, e % hidden_size))

        component_save_path = os.path.join(clusters_dir, f"{c_id}")
        np.save(component_save_path, component)
        print(f"Saved component of {len(component)} neurons with id {c_id} to {component_save_path}.npy")

        if calculate_significance:

            # Get the indices of the significant neurons
            component_indices = [i[0] * hidden_size + i[1] for i in component]

            # If there are no significant neurons, continue to the next component
            if len(component_indices) == 0:
                continue

            component_activations = activations_df.iloc[component_indices].T

            task_labels = np.array([0] * (activations_df.shape[1] // 2) + [1] * (activations_df.shape[1] // 2))

            component_activations = component_activations.to_numpy()

            # linear regression
            lr = LinearRegression()
            lr.fit(component_activations, task_labels)

            pred_r2 = lr.predict(component_activations)

            r2_score_value = r2_score(pred_r2, task_labels)

            print(f"R2 score of component {c_id} is {r2_score_value}")



def generate_bitmap_animation(input_prompt, activation_dir="/content/drive/MyDrive/llm/activations/",
                              output_dir="/content/drive/MyDrive/llm/visualizations/"):
    input_prompt = input_prompt.replace(' ', '_')
    filepath = activation_dir + input_prompt + ".pt"

    # Load the .pt file
    data = torch.load(filepath)

    hidden_states = data['hidden_states']
    output_response = data['output'].split("Response:")[1]

    all_images = []
    vmin = 0
    vmax = 0

    for token_id, token_hidden_states in tqdm(enumerate(hidden_states)):
        if token_id > 0:
            activations = []
            for layer_id, layers in enumerate(token_hidden_states):
                for beam_id, beams in enumerate(layers):
                    for token_activation_id, token_activations in enumerate(beams):
                        token_activations_np = token_activations.cpu().numpy()
                        activations.extend(token_activations_np)

            image_size = int(np.ceil(np.sqrt(len(activations))))
            img = np.zeros((image_size, image_size), dtype=np.uint8)
            img.flat[:len(activations)] = activations
            all_images.append(img)

            vmin_token = np.min(activations)
            if vmin_token < vmin:
                vmin = vmin_token
            vmax_token = np.max(activations)
            if vmax_token > vmax:
                vmax = vmax_token

    log_norm = LogNorm(vmin=vmin, vmax=vmax)

    matplotlib.use("Agg")

    def update(frame):
        plt.clf()
        plt.imshow(all_images[frame], cmap='viridis', norm=log_norm)
        plt.title(f"Prompt: {input_prompt.replace('_', ' ')}, \nToken {frame + 1} Output Token: {output_response[frame]}", fontsize=30)

        plt.axis('off')


    fig = plt.figure(figsize=(20, 20))
    ani = FuncAnimation(fig, update, frames=len(all_images), interval=400)

    output_path = output_dir + input_prompt + ".mp4"
    ani.save(output_path, dpi=100, writer="ffmpeg")

    matplotlib.use("module://ipykernel.pylab.backend_inline")

    print("visualization succesfully saved: '" + str(output_path) + "'")
    