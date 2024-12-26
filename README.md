In this project, a vision-language model is trained using the provided datasets in three steps.
First, a vision encoder is trained on the CIFAR-10 dataset. 
Next, a text decoder is build and trained on the ELI5 dataset. 
Finally, the vision encoder from the first step with the text decoder from the second step are combined to create a vision-language model, 
which is further trained using the provided visual instruction tuning dataset (instruct_tuning.zip).
A minimum baseline performance of 23.6 must be surpassed (<23.6).
