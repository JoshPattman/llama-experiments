import random
from transformers import LlamaTokenizer
from datasets import Dataset
import copy

# Adds either 'a' or 'an' to the start of a string, depending on whether the string starts with a vowel or not
def add_a_or_an(thing:str) ->str:
    if thing[0].lower() in 'aeiou':
        return 'an ' + thing
    else:
        return 'a ' + thing

# Formats a list of strings into a single string, with the last item preceded by 'and' (eg a, b, c and d)
def format_list(things: list[str]) -> str:
    if len(things) == 0:
        return ''
    elif len(things) == 1:
        return things[0]
    elif len(things) == 2:
        return f"{things[0]} and {things[1]}"
    else:
        return ', '.join(things[:-1]) + f", and {things[-1]}"

# Creates a randomised chat prompt given a list of people that take part in the chat. speakers is a list of their tags (eg user, ai) and speaker_full_names is a list of their full names (eg Josh Pattman, Mr Mainframe)
def get_chat_prompt(speakers :list[str], speaker_full_names:list[str], randomise_order:bool = True) -> str:
    if randomise_order and random.randint(0,1) == 0:
        speakers = [speakers[1], speakers[0]]
        speaker_full_names = [speaker_full_names[1], speaker_full_names[0]]
    full_speaker_descs = [f"{speaker_full_names[i]} ({speakers[i]})" for i in range(len(speakers))]
    formatted_speaker_descs = format_list(full_speaker_descs)
    random_choice = random.randint(0,9)
    if random_choice == 0:
        return f"Below is a conversation between {formatted_speaker_descs}:"
    elif random_choice == 1:
        return f"What follows is a conversation between {formatted_speaker_descs}:"
    elif random_choice == 2:
        return f"Here is a conversation between {formatted_speaker_descs}:"
    elif random_choice == 3:
        return f"The following is a discussion between {formatted_speaker_descs}:"
    elif random_choice == 4:
        return f"Here is a dialogue between {formatted_speaker_descs}:"
    elif random_choice == 5:
        return f"Below is a dialogue between {formatted_speaker_descs}:"
    elif random_choice == 6:
        return f"Below is a discussion between {formatted_speaker_descs}:"
    elif random_choice == 7:
        return f"Here is a discussion between {formatted_speaker_descs}:"
    elif random_choice == 8:
        return f"What follows is a dialogue between {formatted_speaker_descs}:"
    elif random_choice == 9:
        return f"The following is a chat between {formatted_speaker_descs}:"

# Formats a conversation into a single string, with each turn preceded by the speaker's name and a colon (eg This is a prompt:\nJosh: Hello</s>Mr Mainframe: Hi Josh</s>)
def format_conversation(prompt: str, conv: list[tuple[str, str]], stop_token: str = "</s>", next_turn: str = "") -> str:
    base = prompt + "\n" + stop_token.join([f"{speaker}: {turn}" for speaker, turn in conv]) + stop_token
    if next_turn != "":
        base += f"{next_turn}:"
    return base

# Takes an initial prompt and a conversation, and formates then tokenizes them for training or inference. It will truncate the conversation if it is too long to fit in the model's max length.
# If for_inference is True, it will return a PyTorch tensor with the input_ids and attention_mask on the GPU. It will also not pad the input_ids. Finally it will truncate from the front not the end.
def tokenize_with_turn_trucation(tokenizer: LlamaTokenizer, prompt:str, conv: list[tuple[str, str]], next_turn: str = "", for_inference=False) -> dict:
    while True:
        txt = format_conversation(prompt, conv, stop_token=tokenizer.eos_token, next_turn=next_turn)
        if for_inference:
            toks = tokenizer(txt, return_tensors="pt").to("cuda")
        else:
            toks = tokenizer(txt, padding="max_length")
        if len(toks["input_ids"]) <= tokenizer.model_max_length:
            if not for_inference:
                toks["text"] = txt
            toks["labels"] = copy.deepcopy(toks["input_ids"])
            return toks
        else:
            if not for_inference:
                conv = conv[:-1]
            else:
                conv = conv[1:]