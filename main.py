import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import gradio as gr

set_seed(67)

device = "mps"

# Initialize models and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
draft_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct", torch_dtype=torch.bfloat16).to(device)
verify_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", torch_dtype=torch.bfloat16).to(device)

def draft(input_ids, gamma, confidence_threshold, eos_token, past_kv):
    generated = input_ids.clone()
    draft_probs = []
    for _ in range(gamma):
        with torch.no_grad():
            outputs = draft_model(
                generated if past_kv is None else generated[:, -1:],
                past_key_values=past_kv,
                use_cache=True
            )
            logits = outputs.logits[:, -1, :]
            past_kv = outputs.past_key_values

        probs = torch.softmax(logits, dim=-1)

        confidence = probs.max().item()
        if confidence < confidence_threshold and len(draft_probs) > 0:
            break

        next_token = torch.argmax(probs, dim=-1, keepdim=True)

        draft_probs.append(probs)
        generated = torch.cat([generated, next_token], dim=-1)

        if next_token.item() == eos_token:
            break

    return generated, draft_probs, past_kv

def verify(drafted, drafted_probs, eos_token, past_kv):
    draft_len = len(drafted_probs)
    with torch.no_grad():
        if past_kv is None:
            target_outputs = verify_model(drafted, use_cache=True)
            target_logits = target_outputs.logits[:, -draft_len - 1:-1, :]
        else:
            target_outputs = verify_model(
                drafted[:, -(draft_len + 1):],
                past_key_values=past_kv,
                use_cache=True
            )
            target_logits = target_outputs.logits[:, :-1, :]

        past_kv = target_outputs.past_key_values

    target_probs = torch.softmax(target_logits, dim=-1)
    accepted_tokens = []
    num_accepted = 0
    for i in range(draft_len):
        q = drafted_probs[i]
        p = target_probs[:, i, :]
        token = drafted[:, i - draft_len]
        x = token[0].item()
        q_x = q[0, x].item()
        p_x = p[0, x].item()

        if q_x <= p_x:
            accepted_tokens.append(x)
            num_accepted += 1
        else:
            r = torch.rand(1).item()
            acceptance_rate = p_x / q_x

            if r < acceptance_rate:
                accepted_tokens.append(x)
                num_accepted += 1
            else:
                adjusted = torch.clamp(p - q, min=0)
                adjusted = adjusted / adjusted.sum()
                new_token = torch.multinomial(adjusted, num_samples=1)[0].item()
                accepted_tokens.append(new_token)
                break
        if accepted_tokens[-1] == eos_token:
            break

    return accepted_tokens, num_accepted, past_kv

def generate_visual(prompt, max_tokens=50, gamma=15, confidence_threshold=0.5):
    # Prepare input
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    eos_token = tokenizer.eos_token_id
    im_end_token = tokenizer.convert_tokens_to_ids("<|im_end|>")

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    result = inputs["input_ids"].clone()

    draft_kv = None
    verify_kv = None

    total_drafted = 0
    total_accepted = 0

    steps = []

    while result.shape[-1] - inputs["input_ids"].shape[-1] < max_tokens:
        print(steps)
        drafted, drafted_probs, draft_kv = draft(result, gamma, confidence_threshold, eos_token, draft_kv)
        accepted_tokens, num_accepted, verify_kv = verify(drafted, drafted_probs, eos_token, verify_kv)

        total_drafted += len(drafted_probs)
        total_accepted += num_accepted

        # Extract token IDs for visualization
        drafted_token_ids = drafted[0, -len(drafted_probs):].tolist()

        step = {
            "drafted": [tokenizer.decode([t]) for t in drafted_token_ids],
            "accepted": num_accepted,
            "resampled": tokenizer.decode([accepted_tokens[-1]]) if num_accepted < len(accepted_tokens) else None
        }
        steps.append(step)

        valid_len = result.shape[-1] + num_accepted
        result = torch.cat([result, torch.tensor([accepted_tokens], device=device)], dim=-1)

        if draft_kv is not None:
            draft_kv.crop(max_length=valid_len)
        if verify_kv is not None:
            verify_kv.crop(max_length=valid_len)

        if eos_token in accepted_tokens or im_end_token in accepted_tokens:
            break
    
    # Extract final output
    final_output = tokenizer.decode(result[0])

    # Build HTML visualization
    html = "<div style='font-family: monospace;'>"
    html += f"<div style='margin-bottom: 20px; padding: 10px; background: transparent; border: 2px solid white; border-radius: 5px;'>"
    html += f"<b>Final Output:</b><br/>{final_output}"
    html += "</div>"
    html += f"<div style='margin-bottom: 20px; padding: 10px; background: transparent; border: 2pd solid white; border-radius: 5px;'>"
    html += f"<b>Acceptance Rate:</b> {total_accepted}/{total_drafted} = {total_accepted/total_drafted*100:.1f}%"
    html += "</div>"
    html += "<div style='margin-bottom: 10px;'><b>Decoding Steps:</b></div>"

    for i, step in enumerate(steps):
        html += f"<div style='margin: 10px 0; padding: 10px; border: 1px solid #ccc; border-radius: 5px;'>"
        html += f"<b>Step {i+1}:</b> "

        for j, token in enumerate(step["drafted"]):
            # Escape HTML special characters
            token_display = token.replace("<", "&lt;").replace(">", "&gt;")
            if j < step["accepted"]:
                html += f"<span style='background: #66CC66; padding: 2px 4px; margin: 2px; border-radius: 3px;'>{token_display}</span>"
            else:
                html += f"<span style='background: #FF8B9A; padding: 2px 4px; margin: 2px; text-decoration: line-through; border-radius: 3px;'>{token_display}</span>"

        if step["resampled"]:
            resampled_display = step["resampled"].replace("<", "&lt;").replace(">", "&gt;")
            html += f" → <span style='background: #5AADCC; padding: 2px 4px; border-radius: 3px;'>{resampled_display}</span>"

        html += "</div>"
    html += "</div>"

    return html

demo = gr.Interface(
    fn=generate_visual,
    inputs=[
        gr.Textbox(label="Prompt", value="What is a deal flow in a VC fund?", lines=3),
        gr.Slider(minimum=10, maximum=100, value=50, step=10, label="Max Tokens"),
        gr.Slider(minimum=1, maximum=30, value=15, step=1, label="Gamma (draft lookahead)"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="Confidence Threshold")
    ],
    outputs=gr.HTML(label="Speculative Decoding Visualization"),
    title="🚀 Speculative Decoding Demo",
    description="""
    **Speculative Decoding Visualization** using Qwen2.5-Coder models

    - **Draft Model**: Qwen2.5-Coder-0.5B-Instruct (fast)
    - **Verify Model**: Qwen2.5-Coder-3B-Instruct (accurate)

    **Color Legend:**
    - 🟢 Green = Accepted tokens from draft model
    - 🔴 Red = Rejected tokens (with strikethrough)
    - 🔵 Blue = Resampled tokens from verify model
    """,
    examples=[
        ["What is a deal flow in a VC fund?", 80, 15, 0.5],
        ["def fibonacci(n):", 50, 15, 0.5],
        ["Explain the concept of attention in transformers", 60, 10, 0.6]
    ]
)

if __name__ == "__main__":
    demo.launch()