import matplotlib.pyplot as plt
import numpy as np

token_counter = 0

task = "classify"

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

            token = line.split(": ")[1]
            
            ymin, ymax = -6, 6

            fig, axs = plt.subplots(31, 1, figsize=(10, 10))
            plt.suptitle(token)
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



