import sys
import discord

import formatting
import json
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from datasets import Dataset
import copy
from transformers import TrainerCallback
from contextlib import nullcontext
from transformers import default_data_collator, Trainer, TrainingArguments
from peft import PeftModel

# LOAD MODEL

# The path to the hugging face model. See the README to get this model.
hugging_face_model_dir = "../../models/llama/7B-hf"
# The path to the trained model. This is generated from the hugging face model train.ipynb
# This file does not include all weights, but simply a small subset of weights that were changed during training.
tuned_model_dir = "./trained-models/llama-7B-v2.1-stanford"

# Load and setup the tokenizer
tokenizer:LlamaTokenizer = LlamaTokenizer.from_pretrained(hugging_face_model_dir)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 256
# Load the base model
model:LlamaForCausalLM = LlamaForCausalLM.from_pretrained(hugging_face_model_dir, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)

# Ontop of the base model, load the modified weights from fine-tuning
model = PeftModel.from_pretrained(model, tuned_model_dir)

# Set the model to evaluation mode
model.eval()

USER_NAME="Josh"
BOT_NAME="Jimmy"

# Get the discord key from the environment variables
DISCORD_KEY = sys.argv[1]
print("running discord bot with key: " + DISCORD_KEY)

conversations = {}

# Setup discord bot
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

def get_reply(conversation) -> str:
    prompt = formatting.get_chat_prompt([USER_NAME, BOT_NAME])
    model_input = formatting.tokenize_with_turn_trucation(tokenizer, prompt, conversation, next_turn=BOT_NAME, for_inference=True)
    with torch.no_grad():
        generated = model.generate(**model_input, max_new_tokens=50, num_beams=1, do_sample=True, temperature=1.2)[0]
        model_output = tokenizer.decode(generated)
    # Parse the model output to get the speaker and what they said
        turns = model_output.split("</s>")
    if turns[-1] == "":
        turns = turns[:-1]
    new_turn = turns[-1].strip()
    new_turn_parts = new_turn.split(": ", 1)
    # Add the new turn to the conversation
    return new_turn_parts[1]

def get_conv_id(message) -> str:
    return f"{message.channel.id}:{message.author.id}"

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    conv_id = get_conv_id(message)
    if message.author == client.user:
        return
    elif message.content == "$shut up":
        if conv_id in conversations:
            del conversations[conv_id]
            await message.channel.send("Didnt want to talk 2 u anyway >:(")
            print("Bot has been silenced")
    elif message.content.startswith('$'):
        reply = 'Bot would say somthing cool here'
        conv = [(USER_NAME, message.content[1:])]
        reply = get_reply(conv)
        conv.append((BOT_NAME, reply))
        conversations[conv_id] = conv
        await message.channel.send(reply)
        print("Sent message: " + reply)
        print(conversations)
    
    elif conv_id in conversations:
        conv = conversations[conv_id]
        conv.append((USER_NAME, message.content))
        reply = get_reply(conv)
        conv.append((BOT_NAME, reply))
        conversations[conv_id] = conv
        await message.channel.send(reply)
        print("Sent message (already existing): " + reply)
        print(conversations)

client.run(DISCORD_KEY)