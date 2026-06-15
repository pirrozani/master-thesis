"""Inference script for address extraction."""

import argparse
import sys
from pathlib import Path

from src.address.inference import AddressExtractor
from src.address.models import Address
from src.config import get_model_config
from src.utils import save_file


def format_address_output(address: Address) -> str:
    """Format address for display.

    Args:
        address: Address object to format

    Returns:
        Formatted string representation
    """

    import json

    json_repr = json.dumps(address.model_dump(), indent=2)
    lines = ['Result:', f'{json_repr}']

    return '\n'.join(lines)


def run_interactive_mode(extractor: AddressExtractor) -> None:
    """Run an interactive chat session for address extraction.

    Args:
        extractor: Initialized AddressExtractor with loaded model
    """
    # Display welcome message
    print('\n' + '=' * 60)
    print('Interactive Address Extraction Mode')
    print('=' * 60)
    print('\nAvailable commands:')
    print('  exit, quit, q    - Exit the session')
    print('  help, ?          - Display this help message')
    print('  multiline, ml    - Enter multi-line input mode')
    print('\nEnter text containing an address to extract, or a command.')
    print('=' * 60 + '\n')

    # Main REPL loop
    running = True
    while running:
        try:
            # Display prompt and get input
            user_input = input('> ').strip()

            # Handle empty input - ignore and re-prompt
            if not user_input:
                continue

            # Check for commands (case-insensitive)
            command = user_input.lower()

            # Handle exit commands
            if command in ['exit', 'quit', 'q']:
                print('\nGoodbye!')
                running = False
                continue

            # Handle help commands
            if command in ['help', '?']:
                print('\nAvailable commands:')
                print('  exit, quit, q    - Exit the session')
                print('  help, ?          - Display this help message')
                print('  multiline, ml    - Enter multi-line input mode')
                print('\nEnter text containing an address to extract.\n')
                continue

            # Handle multiline commands
            if command in ['multiline', 'ml']:
                print('\nEntering multi-line mode. Enter an empty line to finish.\n')
                lines = []
                while True:
                    line = input('... ')
                    if line.strip() == '':
                        break
                    lines.append(line)

                # Concatenate all lines with newlines
                user_input = '\n'.join(lines)

                # If no lines were entered, skip the extraction
                if not user_input.strip():
                    continue

            # Extract address using the extractor
            try:
                address = extractor.extract(user_input)

                # Display formatted output
                print()
                print(format_address_output(address))
                print()

            except RuntimeError as e:
                print(f'\nError: {e}')
                print('Please try again.\n')

        except KeyboardInterrupt:
            print('\n\nInterrupted. Exiting...')
            break
        except EOFError:
            print('\n\nGoodbye!')
            break


def main():
    """Run inference pipeline."""
    parser = argparse.ArgumentParser(
        description='Extract addresses from text using fine-tuned model'
    )
    parser.add_argument(
        'input',
        type=str,
        nargs='?',  # Make input optional for interactive mode
        help='Text input or path to file containing text (one address per line)',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='qwen-3b',
        help='Model alias (e.g., gemma-270m, qwen-3b)',
    )
    parser.add_argument(
        '--adapter-path',
        type=str,
        default=None,
        help='Path to adapter weights directory (overrides model config)',
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Treat input as file path and process multiple lines in batch',
    )
    parser.add_argument(
        '--interactive',
        '-i',
        action='store_true',
        help='Launch interactive chat mode for continuous address extraction',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for inference (cuda or cpu)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: print to stdout)',
    )

    args = parser.parse_args()

    # Validate mutually exclusive flags
    if args.interactive and args.batch:
        print(
            'Error: --interactive and --batch are mutually exclusive',
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate input argument for non-interactive modes
    if not args.interactive and not args.input:
        print(
            'Error: input argument is required for non-interactive mode',
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve model configuration
    config = get_model_config(args.model, task='address')

    # Resolve adapter path (CLI arg overrides config)
    adapter_path = (
        Path(args.adapter_path) if args.adapter_path else Path(config.adapter_dir)
    )
    if not adapter_path.exists():
        print(f'Error: Adapter path does not exist: {adapter_path}', file=sys.stderr)
        sys.exit(1)

    print('\nModel Configuration:')
    print(f'  Model Alias: {args.model}')
    print(f'  Base Model: {config.base_model}')
    print(f'  Adapter Path: {adapter_path}')

    # Initialize extractor
    print(f'\nInitializing AddressExtractor with adapter: {adapter_path}')
    extractor = AddressExtractor(
        model=config, adapter_path=str(adapter_path), device=args.device
    )

    # Load model
    print('Loading model...')
    try:
        extractor.load_model()
    except RuntimeError as e:
        print(f'Error loading model: {e}', file=sys.stderr)
        sys.exit(1)

    # Mode selection: interactive, batch, or single
    if args.interactive:
        # Interactive mode
        run_interactive_mode(extractor)
        return

    # Process input for batch or single mode
    if args.batch:
        # Batch mode: read from a file
        input_path = Path(args.input)
        if not input_path.exists():
            print(f'Error: Input file does not exist: {input_path}', file=sys.stderr)
            sys.exit(1)

        print(f'Reading inputs from {input_path}...')
        with open(input_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

        if not texts:
            print('Error: No valid input lines found in file', file=sys.stderr)
            sys.exit(1)

        print(f'Processing {len(texts)} inputs in batch...\n')
        addresses = extractor.extract_batch(texts)

        # Format output
        results = []
        for i, (text, address) in enumerate(zip(texts, addresses), 1):
            results.append(f'Input {i}: {text}')
            results.append(format_address_output(address))
            results.append('')  # Empty line between results

    else:
        # Single mode: treat input as text or file
        input_path = Path(args.input)

        if input_path.exists():
            # Input is a file path
            print(f'Reading input from {input_path}...')
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        else:
            # Input is raw text
            text = args.input

        if not text:
            print('Error: Empty input provided', file=sys.stderr)
            sys.exit(1)

        print(f'Processing input: {text}\n')
        address = extractor.extract(text)

        # Format output
        results = [f'Input: {text}', format_address_output(address)]

    # Output results
    output_text = '\n'.join(results)

    if args.output:
        save_file(output_text, args.output)
        print(f'\nResults saved to {args.output}')
    else:
        print('=' * 60)
        print(output_text)
        print('=' * 60)


if __name__ == '__main__':
    main()
