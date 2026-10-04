import argparse
import torch
from src.model import TransformerLM
from src.checkpoint import load_checkpoint
from src.tokenizer import Tokenizer
from src.decoding import generate

def main():
    parser = argparse.ArgumentParser(description="Language Model Text Generation")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/tiny_model.pt")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda")
    
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1024)
    
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    tokenizer = Tokenizer.from_files(
        vocab_filepath="data/tokenizer_vocab.pkl", 
        merges_filepath="data/tokenizer_merges.pkl"
    )

    model = TransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
        ).to(device)

    load_checkpoint(args.checkpoint_path, model)
    print(f"Model loaded from: {args.checkpoint_path}")

    prompt_ids = torch.tensor(tokenizer.encode(args.prompt), dtype=torch.long, device=device).unsqueeze(0)

    output_ids = generate(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        context_length=args.context_length,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=None
    )

    generated_text = tokenizer.decode(output_ids[0].tolist())

    print(generated_text)

if __name__ == "__main__":
    main()