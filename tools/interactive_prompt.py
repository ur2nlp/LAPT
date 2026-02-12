"""
Interactive prompt script for testing trained language models.

Usage:
    python tools/interactive_prompt.py --model path/to/model
    python tools/interactive_prompt.py --model models/got/l40-basic --temperature 0.9 --max-tokens 200
    python tools/interactive_prompt.py --model models/got/l40-basic --device cuda

Commands (during generation):
    /temp <value>     - Set temperature
    /topp <value>     - Set top-p nucleus sampling threshold
    /max <tokens>     - Set maximum number of tokens to generate
    /rep <value>      - Set repetition penalty (1.0 = off)
    /nogram <size>    - Set no-repeat n-gram size (0 = off)
    /sample           - Enable sampling
    /nosample         - Disable sampling (greedy decoding)
    /settings         - Show current generation settings
    /help             - Show available commands
"""

import argparse
import torch
from transformers import pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Interactive text generation with a trained model"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to the trained model directory'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature (default: 0.7)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=100,
        help='Maximum number of new tokens to generate (default: 100)'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.9,
        help='Nucleus sampling probability (default: 0.9)'
    )
    parser.add_argument(
        '--repetition-penalty',
        type=float,
        default=1.0,
        help='Repetition penalty (1.0 = no penalty, >1.0 penalizes repetition, default: 1.0)'
    )
    parser.add_argument(
        '--no-repeat-ngram-size',
        type=int,
        default=0,
        help='Prevent repeating any n-gram of this size (0 = off, default: 0)'
    )
    parser.add_argument(
        '--no-sample',
        action='store_true',
        help='Use greedy decoding instead of sampling'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use: "auto" (default), "cuda", "cpu", or device number like "cuda:0"'
    )

    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        device = 0 if torch.cuda.is_available() else -1
        device_name = f"cuda:{device}" if device >= 0 else "cpu"
    elif args.device == 'cpu':
        device = -1
        device_name = "cpu"
    elif args.device == 'cuda' or args.device.startswith('cuda:'):
        device = 0 if args.device == 'cuda' else int(args.device.split(':')[1])
        device_name = f"cuda:{device}"
    else:
        device = int(args.device)
        device_name = f"cuda:{device}"

    print(f"Loading model from {args.model}...")
    print(f"Using device: {device_name}")
    generator = pipeline('text-generation', model=args.model, device=device)

    def show_settings():
        """Display current generation settings."""
        print("\nCurrent generation settings:")
        print(f"  Device: {device_name}")
        print(f"  Temperature: {args.temperature}")
        print(f"  Max new tokens: {args.max_tokens}")
        print(f"  Top-p: {args.top_p}")
        print(f"  Repetition penalty: {args.repetition_penalty}")
        print(f"  No-repeat n-gram size: {args.no_repeat_ngram_size}")
        print(f"  Sampling: {'disabled (greedy)' if args.no_sample else 'enabled'}")
        print()

    show_settings()
    print("Type 'quit', 'exit', or 'q' to exit.")
    print("Type '/help' for available commands.\n")

    while True:
        try:
            prompt = input("Prompt: ")

            # Check for exit commands
            if prompt.lower().strip() in ['quit', 'exit', 'q', '']:
                break

            # Check for configuration commands
            if prompt.startswith('/'):
                parts = prompt.split()
                cmd = parts[0][1:].lower()

                if cmd == 'temp' or cmd == 'temperature':
                    if len(parts) < 2:
                        print("Usage: /temp <value>")
                    else:
                        try:
                            args.temperature = float(parts[1])
                            print(f"Temperature set to {args.temperature}")
                        except ValueError:
                            print(f"Invalid temperature value: {parts[1]}")

                elif cmd == 'topp' or cmd == 'top-p':
                    if len(parts) < 2:
                        print("Usage: /topp <value>")
                    else:
                        try:
                            args.top_p = float(parts[1])
                            print(f"Top-p set to {args.top_p}")
                        except ValueError:
                            print(f"Invalid top-p value: {parts[1]}")

                elif cmd == 'max' or cmd == 'max-tokens':
                    if len(parts) < 2:
                        print("Usage: /max <tokens>")
                    else:
                        try:
                            args.max_tokens = int(parts[1])
                            print(f"Max tokens set to {args.max_tokens}")
                        except ValueError:
                            print(f"Invalid token count: {parts[1]}")

                elif cmd == 'rep' or cmd == 'repetition-penalty':
                    if len(parts) < 2:
                        print("Usage: /rep <value>  (1.0 = off, try 1.1-1.3)")
                    else:
                        try:
                            args.repetition_penalty = float(parts[1])
                            print(f"Repetition penalty set to {args.repetition_penalty}")
                        except ValueError:
                            print(f"Invalid repetition penalty value: {parts[1]}")

                elif cmd == 'nogram' or cmd == 'no-repeat-ngram':
                    if len(parts) < 2:
                        print("Usage: /nogram <size>  (0 = off, try 3-4)")
                    else:
                        try:
                            args.no_repeat_ngram_size = int(parts[1])
                            print(f"No-repeat n-gram size set to {args.no_repeat_ngram_size}")
                        except ValueError:
                            print(f"Invalid n-gram size: {parts[1]}")

                elif cmd == 'sample':
                    args.no_sample = False
                    print("Sampling enabled")

                elif cmd == 'nosample' or cmd == 'greedy':
                    args.no_sample = True
                    print("Sampling disabled (greedy decoding)")

                elif cmd == 'settings' or cmd == 'config':
                    show_settings()

                elif cmd == 'help' or cmd == 'h':
                    print("\nAvailable commands:")
                    print("  /temp <value>     - Set temperature")
                    print("  /topp <value>     - Set top-p nucleus sampling threshold")
                    print("  /max <tokens>     - Set maximum number of tokens to generate")
                    print("  /rep <value>      - Set repetition penalty (1.0 = off, try 1.1-1.3)")
                    print("  /nogram <size>    - Set no-repeat n-gram size (0 = off, try 3-4)")
                    print("  /sample           - Enable sampling")
                    print("  /nosample         - Disable sampling (greedy decoding)")
                    print("  /settings         - Show current generation settings")
                    print("  /help             - Show this help message")
                    print()

                else:
                    print(f"Unknown command: /{cmd}. Type /help for available commands.")

                continue

            # Generate text
            generate_kwargs = dict(
                max_new_tokens=args.max_tokens,
                do_sample=not args.no_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=generator.tokenizer.eos_token_id,
            )
            if args.repetition_penalty != 1.0:
                generate_kwargs['repetition_penalty'] = args.repetition_penalty
            if args.no_repeat_ngram_size > 0:
                generate_kwargs['no_repeat_ngram_size'] = args.no_repeat_ngram_size

            output = generator(prompt, **generate_kwargs)

            print(f"\n{output[0]['generated_text']}\n")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == '__main__':
    main()
