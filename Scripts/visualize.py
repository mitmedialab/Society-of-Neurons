import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import tqdm
import os
from PyDictionary import PyDictionary

from matplotlib import cm
from matplotlib.colors import LightSource

from transformers import LlamaTokenizer

d = PyDictionary()

tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf')

activation_file = 'activations/What_is_a_llama_.pt'
# activation_file = 'activations/Pretend_you_are_Einstein,_are_you_a_male_or_a_female_.pt'
# activation_file = 'activations/Pretend_you_are_Curie,_are_you_a_male_or_a_female_.pt'
# activation_file = 'activations/What_is_5_+_9_.pt'

generation_output = torch.load(activation_file, map_location=torch.device('cpu'))

# get the activations
prompt = generation_output['prompt']
hidden_states = generation_output['hidden_states']
output_sequence = generation_output['sequences'][0]

output = generation_output['output'].split("Response:")[1]
output_token_ids = tokenizer(output, return_tensors="pt")['input_ids'][0]

# output_token_ids = tokenizer.encode(output)
# output = ""
# for i in output_token_ids:
#     output += tokenizer.decode([i])+" "
# print(output)
# exit()

assert len(output_token_ids) == len(hidden_states), "output token ids and hidden states don't match"

activations = []

for i in range(len(hidden_states)):
    activations.append(hidden_states[i])

activations = activations[1:]
output_token_ids = output_token_ids[1:]

def plot_activations(prompt, activations, suptitle, output_token_id, token_id):
    # plot the activations
    n_hidden_layers = len(activations)

    fig, axs = plt.subplots(n_hidden_layers, 1, figsize=(10, 12))

    output_token = tokenizer.decode(output_token_id)

    if len(suptitle.split(' ')) % 30 == 0:
        suptitle += "\n"

    # decode output token and add to suptitle
    suptitle += output_token.replace("<0x0A>", "\n") + " "
    plt.suptitle(suptitle, fontsize=8)

    for j in range(n_hidden_layers):
        layer_activations = activations[j].numpy().squeeze(0).squeeze(0)
        axs[j].plot(layer_activations)
        # axs[j].set_title(f"Layer {j}", f ontsize=2)
        axs[j].set_xticks([])
        axs[j].set_xticklabels([])
        axs[j].set_yticks([])
        axs[j].set_yticklabels([])

    fig.text(0.5, 0.04, 'Embedding: 1 (left) - 4096 (right)', ha='center')
    fig.text(0.04, 0.5, 'Layers: 1 (top) - 32 (bottom)', va='center', rotation='vertical')

    # fig.tight_layout()
    # save the plot
    plt.savefig("activations/plots/{}/{}_{}.png"
    .format(
        prompt.replace(" ", "_"),
        prompt.replace(" ", "_"), 
        str(token_id).zfill(4)))

    return suptitle

print("plotting activations")

# make a folder for the plots
os.makedirs("activations/plots/{}".format(prompt.replace(" ", "_")), exist_ok=True)

suptitle = prompt
suptitle += "\n\n\n"

for i in tqdm.tqdm(range(len(activations))):
    suptitle = plot_activations(
        prompt,
        activations[i], 
        suptitle, 
        output_token_ids[i],
        i)

print("Making video")

import cv2
path = "activations/plots/{}".format(prompt.replace(" ", "_"))

# Get list of all image files in the folder
images = [img for img in os.listdir(path) if img.endswith(".png")]

# Sort the images by filename
images.sort()

# Get image dimensions
img = cv2.imread(os.path.join(path, images[0]))
height, width, channels = img.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("activations/videos/{}.mp4".format(prompt.replace(" ", "_")),
 fourcc, 5, (width, height))

# Loop through the images and write each frame to the video
for image in images:
    img = cv2.imread(os.path.join(path, image))
    out.write(img)

# Release the VideoWriter object
out.release()