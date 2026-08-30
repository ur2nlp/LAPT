"""
Interactive prompt script for testing trained language models.

Usage:
    python tools/interactive_prompt.py --model path/to/model
    python tools/interactive_prompt.py --model models/got/l40-basic --temperature 0.9 --max-tokens 200
    python tools/interactive_prompt.py --model models/got/l40-basic --device cuda
    python tools/interactive_prompt.py --model models/got/l40-basic --compile
    python tools/interactive_prompt.py --model models/got/l40-basic --template templates/translate_got.txt
    python tools/interactive_prompt.py --model models/got/l40-basic --batch prompts.txt

Template files:
    A template file is a text file with an {input} placeholder. Everything before {input}
    becomes the prompt prefix, and everything after becomes the prompt suffix. The model
    generates a continuation after the full formatted prompt, and only the new text is shown.

    Example template (templates/translate_got.txt):
        Translate to Gothic: {input} Response:

    With this template, typing "the light of the world" sends:
        "Translate to Gothic: the light of the world Response:"
    and displays only the model's continuation.

Batch mode (--batch):
    Reads prompts from a file, one per line. Blank lines and lines starting with '#' are
    skipped. Each prompt is echoed and the model's response is printed, then the script exits.
    The template (if any) is applied to each prompt, just as in interactive mode.

    To include a newline within a single prompt, write a literal '\n' escape in the line; it
    is decoded to a real newline before the prompt is sent. For example, the batch line
    "Line one\nLine two" sends a two-line prompt.

Commands (during interactive session):
    /temp <value>       - Set temperature
    /topp <value>       - Set top-p nucleus sampling threshold
    /max <tokens>       - Set maximum number of tokens to generate
    /rep <value>        - Set repetition penalty (1.0 = off)
    /nogram <size>      - Set no-repeat n-gram size (0 = off)
    /sample             - Enable sampling
    /nosample           - Disable sampling (greedy decoding)
    /reload <path>      - Load a different model (keeps device/dtype/settings)
    /compile            - Compile the current model with torch.compile
    /template <path>    - Load a prompt template (use {input} placeholder)
    /template off       - Disable the current template
    /template           - Show the current template
    /raw <text>         - Send raw text to the model, bypassing the template
    /fulltext           - Toggle showing full text (prompt + response) vs response only
    /settings           - Show current generation settings
    /help               - Show available commands
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import pipeline


@dataclass
class Session:
    """Mutable state for an interactive prompting session."""
    # Model
    generator: pipeline
    model_path: str
    compiled: bool

    # Device/dtype (fixed for the session)
    device: int
    device_name: str
    model_dtype: torch.dtype

    # Template
    template: str | None = None
    template_path: str | None = None

    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 100
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    do_sample: bool = True

    # Display
    show_full_text: bool = False
    prompt_label: str = ">"


def resolve_device(device_str: str) -> tuple[int, str]:
    """Convert device string to pipeline device int and display name."""
    if device_str == 'auto':
        device = 0 if torch.cuda.is_available() else -1
        device_name = f"cuda:{device}" if device >= 0 else "cpu"
    elif device_str == 'cpu':
        device = -1
        device_name = "cpu"
    elif device_str == 'cuda' or device_str.startswith('cuda:'):
        device = 0 if device_str == 'cuda' else int(device_str.split(':')[1])
        device_name = f"cuda:{device}"
    else:
        device = int(device_str)
        device_name = f"cuda:{device}"
    return device, device_name


def resolve_dtype(dtype_str: str, device: int) -> torch.dtype:
    """Convert dtype string to torch dtype, using device to pick a default."""
    dtype_map = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}
    if dtype_str == 'auto':
        return torch.bfloat16 if device >= 0 else torch.float32
    return dtype_map[dtype_str]


def load_model(
    model_path: str,
    device: int,
    model_dtype: torch.dtype,
    compile_model: bool,
) -> pipeline:
    """Load a text-generation pipeline, optionally with torch.compile."""
    print(f"Loading model from {model_path}...")
    generator = pipeline(
        'text-generation', model=model_path, device=device, torch_dtype=model_dtype,
    )
    if compile_model:
        print("Compiling model with torch.compile...")
        generator.model = torch.compile(generator.model)
        print("Compiled. (First generation will be slow due to tracing.)")
    return generator


def load_template(template_path: str) -> str:
    """Load a template file and validate it contains {input}."""
    path = Path(template_path)
    if not path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    template = path.read_text().strip()
    if '{input}' not in template:
        raise ValueError(
            f"Template must contain {{input}} placeholder, got: {template!r}"
        )
    return template


def apply_template(template: str, user_input: str) -> str:
    """Substitute user input into the template."""
    return template.replace('{input}', user_input)


def show_settings(session: Session):
    """Display current generation settings."""
    print("\nCurrent generation settings:")
    print(f"  Model: {session.model_path}")
    print(f"  Device: {session.device_name}")
    print(f"  Dtype: {session.model_dtype}")
    print(f"  Compiled: {session.compiled}")
    if session.template:
        print(f"  Template: {session.template!r}")
        print(f"    (from {session.template_path})")
    else:
        print("  Template: off")
    print(f"  Temperature: {session.temperature}")
    print(f"  Max new tokens: {session.max_tokens}")
    print(f"  Top-p: {session.top_p}")
    print(f"  Repetition penalty: {session.repetition_penalty}")
    print(f"  No-repeat n-gram size: {session.no_repeat_ngram_size}")
    print(f"  Sampling: {'enabled' if session.do_sample else 'disabled (greedy)'}")
    print(f"  Display: {'full text' if session.show_full_text else 'response only'}")
    print()


def generate_and_print(session: Session, full_prompt: str):
    """Generate text from a prompt and print only the new tokens."""
    generate_kwargs = dict(
        max_new_tokens=session.max_tokens,
        do_sample=session.do_sample,
        temperature=session.temperature,
        top_p=session.top_p,
        # Pass EOS explicitly from the tokenizer so a stale generation_config
        # (e.g. base-model EOS id surviving a vocab swap) cannot silently
        # prevent the model from halting on its real end-of-sequence token.
        eos_token_id=session.generator.tokenizer.eos_token_id,
        pad_token_id=session.generator.tokenizer.eos_token_id,
    )
    if session.repetition_penalty != 1.0:
        generate_kwargs['repetition_penalty'] = session.repetition_penalty
    if session.no_repeat_ngram_size > 0:
        generate_kwargs['no_repeat_ngram_size'] = session.no_repeat_ngram_size

    output = session.generator(full_prompt, **generate_kwargs)
    generated_text = output[0]['generated_text']

    if session.show_full_text:
        print(f"\n{generated_text}\n")
    else:
        response = generated_text[len(full_prompt):]
        print(f"\n{response}\n")


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
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'bf16', 'fp16', 'fp32'],
        help='Model dtype: "auto" (bf16 on CUDA, fp32 on CPU), "bf16", "fp16", or "fp32"'
    )
    parser.add_argument(
        '--compile',
        action='store_true',
        help='Compile model with torch.compile (slower first call, faster subsequent calls)'
    )
    parser.add_argument(
        '--template',
        type=str,
        default=None,
        help='Path to a prompt template file (text file with {input} placeholder)'
    )
    parser.add_argument(
        '--batch',
        type=str,
        default=None,
        metavar='FILE',
        help='Run non-interactively, reading prompts from FILE (one per line; blank lines and # comments skipped)'
    )

    args = parser.parse_args()

    device, device_name = resolve_device(args.device)
    model_dtype = resolve_dtype(args.dtype, device)

    # Load template if provided
    template = None
    template_path = None
    if args.template:
        try:
            template = load_template(args.template)
            template_path = args.template
            print(f"Template loaded: {template!r}")
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: {e}")
            print("Continuing without template.")

    print(f"Using device: {device_name}, dtype: {model_dtype}")
    generator = load_model(args.model, device, model_dtype, args.compile)

    session = Session(
        generator=generator,
        model_path=args.model,
        compiled=args.compile,
        device=device,
        device_name=device_name,
        model_dtype=model_dtype,
        template=template,
        template_path=template_path,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        do_sample=not args.no_sample,
    )

    show_settings(session)

    if args.batch:
        batch_path = Path(args.batch)
        if not batch_path.exists():
            print(f"Error: batch file not found: {args.batch}", file=sys.stderr)
            return
        lines = batch_path.read_text().splitlines()
        prompts = [
            line.replace('\\n', '\n')
            for line in lines
            if line.strip() and not line.strip().startswith('#')
        ]
        print(f"Running batch mode with {len(prompts)} prompt(s) from {args.batch}\n")
        for prompt in prompts:
            print(f"> {prompt}")
            if session.template:
                full_prompt = apply_template(session.template, prompt)
            else:
                full_prompt = prompt
            try:
                generate_and_print(session, full_prompt)
            except Exception as e:
                print(f"Error generating for prompt {prompt!r}: {e}", file=sys.stderr)
        return

    print("Type 'quit', 'exit', or 'q' to exit.")
    print("Type '/help' for available commands.\n")

    while True:
        try:
            user_input = input(f"{session.prompt_label} ")

            # Check for exit commands
            if user_input.lower().strip() in ['quit', 'exit', 'q', '']:
                break

            # Check for configuration commands
            if user_input.startswith('/'):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0][1:].lower()
                rest = parts[1] if len(parts) > 1 else ''

                if cmd == 'temp' or cmd == 'temperature':
                    if not rest:
                        print("Usage: /temp <value>")
                    else:
                        try:
                            session.temperature = float(rest)
                            print(f"Temperature set to {session.temperature}")
                        except ValueError:
                            print(f"Invalid temperature value: {rest}")

                elif cmd == 'topp' or cmd == 'top-p':
                    if not rest:
                        print("Usage: /topp <value>")
                    else:
                        try:
                            session.top_p = float(rest)
                            print(f"Top-p set to {session.top_p}")
                        except ValueError:
                            print(f"Invalid top-p value: {rest}")

                elif cmd == 'max' or cmd == 'max-tokens':
                    if not rest:
                        print("Usage: /max <tokens>")
                    else:
                        try:
                            session.max_tokens = int(rest)
                            print(f"Max tokens set to {session.max_tokens}")
                        except ValueError:
                            print(f"Invalid token count: {rest}")

                elif cmd == 'rep' or cmd == 'repetition-penalty':
                    if not rest:
                        print("Usage: /rep <value>  (1.0 = off, try 1.1-1.3)")
                    else:
                        try:
                            session.repetition_penalty = float(rest)
                            print(f"Repetition penalty set to {session.repetition_penalty}")
                        except ValueError:
                            print(f"Invalid repetition penalty value: {rest}")

                elif cmd == 'nogram' or cmd == 'no-repeat-ngram':
                    if not rest:
                        print("Usage: /nogram <size>  (0 = off, try 3-4)")
                    else:
                        try:
                            session.no_repeat_ngram_size = int(rest)
                            print(f"No-repeat n-gram size set to {session.no_repeat_ngram_size}")
                        except ValueError:
                            print(f"Invalid n-gram size: {rest}")

                elif cmd == 'sample':
                    session.do_sample = True
                    print("Sampling enabled")

                elif cmd == 'nosample' or cmd == 'greedy':
                    session.do_sample = False
                    print("Sampling disabled (greedy decoding)")

                elif cmd == 'reload':
                    if not rest:
                        print("Usage: /reload <path/to/model>")
                    else:
                        try:
                            session.generator = load_model(
                                rest, session.device, session.model_dtype,
                                session.compiled,
                            )
                            session.model_path = rest
                        except Exception as e:
                            print(f"Failed to load model: {e}")
                            print(f"Keeping previous model: {session.model_path}")

                elif cmd == 'compile':
                    if session.compiled:
                        print("Model is already compiled.")
                    else:
                        print("Compiling model with torch.compile...")
                        session.generator.model = torch.compile(
                            session.generator.model
                        )
                        session.compiled = True
                        print("Compiled. (First generation will be slow due to tracing.)")

                elif cmd == 'template':
                    if not rest:
                        if session.template:
                            print(f"Current template: {session.template!r}")
                            print(f"  (from {session.template_path})")
                        else:
                            print("No template active.")
                    elif rest.lower() == 'off':
                        session.template = None
                        session.template_path = None
                        print("Template disabled.")
                    else:
                        try:
                            session.template = load_template(rest)
                            session.template_path = rest
                            print(f"Template loaded: {session.template!r}")
                        except (FileNotFoundError, ValueError) as e:
                            print(f"Error: {e}")

                elif cmd == 'raw':
                    if not rest:
                        print("Usage: /raw <text to send directly to model>")
                    else:
                        generate_and_print(session, rest)

                elif cmd == 'fulltext':
                    session.show_full_text = not session.show_full_text
                    mode = "full text" if session.show_full_text else "response only"
                    print(f"Display mode: {mode}")

                elif cmd == 'prompt':
                    if not rest:
                        print(f"Current prompt label: {session.prompt_label!r}")
                        print("Usage: /prompt <label>  (e.g., /prompt >, /prompt Prompt:)")
                    else:
                        session.prompt_label = rest
                        print(f"Prompt label set to {session.prompt_label!r}")

                elif cmd == 'settings' or cmd == 'config':
                    show_settings(session)

                elif cmd == 'help' or cmd == 'h':
                    print("\nAvailable commands:")
                    print("  /temp <value>       - Set temperature")
                    print("  /topp <value>       - Set top-p nucleus sampling threshold")
                    print("  /max <tokens>       - Set maximum number of tokens to generate")
                    print("  /rep <value>        - Set repetition penalty (1.0 = off, try 1.1-1.3)")
                    print("  /nogram <size>      - Set no-repeat n-gram size (0 = off, try 3-4)")
                    print("  /sample             - Enable sampling")
                    print("  /nosample           - Disable sampling (greedy decoding)")
                    print("  /reload <path>      - Load a different model (keeps settings)")
                    print("  /compile            - Compile model with torch.compile")
                    print("  /template <path>    - Load a prompt template ({input} placeholder)")
                    print("  /template off       - Disable the current template")
                    print("  /template           - Show the current template")
                    print("  /raw <text>         - Send raw text, bypassing the template")
                    print("  /fulltext           - Toggle full text vs response-only display")
                    print("  /prompt <label>     - Change the input prompt label (default: >)")
                    print("  /settings           - Show current generation settings")
                    print("  /help               - Show this help message")
                    print()

                else:
                    print(f"Unknown command: /{cmd}. Type /help for available commands.")

                continue

            # Apply template if active, otherwise use input directly
            if session.template:
                full_prompt = apply_template(session.template, user_input)
            else:
                full_prompt = user_input

            generate_and_print(session, full_prompt)

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == '__main__':
    main()
