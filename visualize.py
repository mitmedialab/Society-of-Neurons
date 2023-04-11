import matplotlib.pyplot as plt
import numpy as np
import sys

token_counter = 0

if(len(sys.argv))<2:
    print("Enter the task")
    exit()

task = sys.argv[1]

all_tokens = []

# mkdir token_activations if it doesn't exist
import os
if not os.path.exists(task+"_token_activations"):
    os.mkdir(task+"_token_activations")


# read a text file and convert it to a list
with open(task+"_activations.txt", 'r') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]

    # check if line contains layer
    for line in lines:
        
        activations_for_token = []
        if "Token" in line:
            print(line)
            # get all the activations in the previous 64 lines and plot them one by one
            for i in range(1, 63, 2):
                activation_line = lines[lines.index(line) - i]
                # convert the line to a list of floats
                activations = activation_line.split(",")[:-1]
                activations = [float(x) for x in activations]
                # plot the activations
                activations_for_token.append(activations)

            if len(line.split(": ")) <2:
                break

            token = line.split(": ")[1]
            all_tokens.append(token)

            if len(all_tokens) %20==0:
                all_tokens.append('\n')
            
            ymin, ymax = -8, 8

            fig, axs = plt.subplots(31, 1, figsize=(10, 10))
            plt.suptitle(''.join(all_tokens))
            for j in range(31):
                axs[j].set_ylim([ymin, ymax])
                axs[j].plot(activations_for_token[j])
                # axs[j].set_title(f"Layer {j}", fontsize=2)
                axs[j].set_xticks([])
                axs[j].set_xticklabels([])
                axs[j].set_yticks([])
                axs[j].set_yticklabels([])

            # fig.tight_layout()
            # save the plot
            # format token_counter to 4 digits int
            plt.savefig(task+f"_token_activations/{str(token_counter).zfill(3)}{token}.png")
            # plt.show()
            token_counter+=1


# make a video with all the figures in task_token_activations

import cv2
path = task+"_token_activations"

# Get list of all image files in the folder
images = [img for img in os.listdir(path) if img.endswith(".png")]

# Sort the images by filename
images.sort()

# Get image dimensions
img = cv2.imread(os.path.join(path, images[0]))
height, width, channels = img.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(task+".mp4", fourcc, 5, (width, height))

# Loop through the images and write each frame to the video
for image in images:
    img = cv2.imread(os.path.join(path, image))
    out.write(img)

# Release the VideoWriter object
out.release()
