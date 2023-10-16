## Downloading the model
First, download the weights from [this link](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform?usp=send_form).

Then, use [this script](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py) to convert the model.

`python convert_llama_weights_to_hf.py --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path$`

This will output a directory that is the hugging face format weights for the model. This is ~14GB.

for full instructions, look at [this webpage](https://huggingface.co/docs/transformers/main/model_doc/llama).