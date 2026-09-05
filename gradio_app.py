import os
# PyTorch ships its own OpenMP runtime, and conda has one too. On macOS both get
# loaded into the same process and OpenMP aborts as a safety measure. This flag
# tells it to continue anyway. Must be set BEFORE torch/transformers are imported.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch


import gradio as gr                                    # builds the web UI
from transformers import AutoModelForCausalLM, AutoTokenizer
# AutoModelForCausalLM -> "predict the next token" models (the GPT family)
# AutoTokenizer        -> converts text into the integer IDs a model reads, and back


def generate_answer(question, model):
    # This checkpoint (Open-Orca/oo-phi-1_5) was fine-tuned on ChatML, not on plain
    # text continuation. It only reliably stops itself if the prompt is wrapped in
    # <|im_start|>role ... <|im_end|> turns — feeding it a bare string makes it treat
    # the prompt as a document to keep extending, which is why it used to wander into
    # unrelated invented content (fake emails, unrelated topics, etc.) once the real
    # answer was done.
    sys_prompt = (
        "I am OrcaPhi, an AI health assistant. I give careful, concise, factual "
        "answers and do not invent unrelated content."
    )
    prompt = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # Loads the tokenizer from the local phi1.5 folder (relative path — this is why
    # the folder must sit next to this script and be spelled exactly "phi1.5").
    tokenizer = AutoTokenizer.from_pretrained("phi1.5")

    # Text -> token IDs. return_tensors="pt" wraps the IDs in a PyTorch tensor
    # of shape [1, sequence_length] rather than a plain Python list.
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Move that tensor onto the same device the model lives on. The weights are on
    # MPS (Apple's GPU), so CPU input would raise "Passed CPU tensor to MPS op".
    # Using model.device instead of hardcoding "mps" keeps this portable.
    input_ids = input_ids.to(model.device)

    # <|im_end|> is this checkpoint's real end-of-turn token (see added_tokens.json).
    # Passing it as eos_token_id lets generate() stop as soon as the model signals
    # it's done, instead of always padding out to the max length.
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # The generation loop: predicts one token, appends it, predicts the next, repeat.
    #   max_new_tokens=256 -> stop after 256 NEW tokens (prompt no longer counts,
    #                         now that the system+user prompt itself is longer)
    #   do_sample=True     -> pick randomly from likely tokens instead of always the
    #                         top one; this is what makes answers vary between runs
    #   top_k=50           -> only ever consider the 50 most likely next tokens
    #   top_p=0.9          -> ...and within those, only until probabilities sum to 0.9
    #   eos_token_id       -> stop generating once <|im_end|> is produced
    # [0] takes the first sequence out of the batch (we only sent one).
    output = model.generate(
        input_ids,
        max_new_tokens=256,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        eos_token_id=im_end_id,
        pad_token_id=tokenizer.pad_token_id,
    )[0]

    # Only decode the newly generated tokens — the input includes the system/user
    # turns, which we don't want echoed back into the answer.
    new_tokens = output[input_ids.shape[-1]:]

    # Token IDs -> readable text. skip_special_tokens drops internal markers
    # like <|im_end|> so they don't appear in the UI.
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return answer.strip()


def chatbot(question):
    # Gradio calls this with whatever the user typed. It just forwards to
    # generate_answer, passing in the model loaded below as a global.
    answer = generate_answer(question, llm_model)
    return answer


if __name__ == "__main__":
    # Load the pre-trained model (outside the interface definition for efficiency)
    # llm_model = AutoModelForCausalLM.from_pretrained("phi1.5", trust_remote_code=True)

    # from_pretrained reads config.json, builds the architecture, loads ~2.8GB of
    # weights. trust_remote_code=True executes the custom modeling_mixformer_
    # sequential.py shipped in the repo — only ever enable this for sources you trust.
    # .to("mps") moves the weights to Apple's GPU. Windows students: delete .to("mps").
    # llm_model = AutoModelForCausalLM.from_pretrained("phi1.5", trust_remote_code=True).to("mps")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    llm_model = AutoModelForCausalLM.from_pretrained("phi1.5", trust_remote_code=True).to(device)

    # Wires the UI: one text input, one text output, both handled by chatbot().
    # Gradio infers the widgets from the "text" strings.
    interface = gr.Interface(
        fn=chatbot,
        inputs="text",
        outputs="text",
        title="I am your AI Health Assistance 🏥",
        description="As general health realted question to the AI Bot."
    )

    # Starts a local web server at http://127.0.0.1:7860 and blocks.
    # Add share=True for a temporary public URL your students can open.
    interface.launch()