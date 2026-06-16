"""Inference script for entity type classification."""

import argparse
import json
import sys
from pathlib import Path

from src.entity_name.classification.inference import EntityTypeClassifier
from src.entity_name.classification.models import EntityType
from src.config import get_model_config, DEFAULT_MODEL
from src.utils import save_file

TASK = 'entity_name/classification'


def format_entity_type_output(entity_type: EntityType) -> str:
    """Format an entity type prediction for display.

    Args:
        entity_type: EntityType object to format

    Returns:
        Formatted string representation
    """
    json_repr = json.dumps(entity_type.model_dump(), indent=2)
    lines = ['Result:', f'{json_repr}']

    return '\n'.join(lines)


def run_interactive_mode(classifier: EntityTypeClassifier) -> None:
    """Run an interactive session for entity type classification.

    Args:
        classifier: Initialized EntityTypeClassifier with loaded model
    """
    # Display welcome message
    print('\n' + '=' * 60)
    print('Interactive Entity Type Classification Mode')
    print('=' * 60)
    print('\nAvailable commands:')
    print('  exit, quit, q    - Exit the session')
    print('  help, ?          - Display this help message')
    print('\nEnter an entity name to classify, or a command.')
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
                print('\nEnter an entity name to classify.\n')
                continue

            # Classify the entity name
            try:
                entity_type = classifier.classify(user_input)

                # Display formatted output
                print()
                print(format_entity_type_output(entity_type))
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
        description='Classify entity names as company or person using fine-tuned model'
    )
    parser.add_argument(
        'input',
        type=str,
        nargs='?',  # Make input optional for interactive mode
        help='Text input or path to file containing names (one per line)',
    )
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL,
        help=f'Model alias (default: {DEFAULT_MODEL})',
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
        help='Launch interactive mode for continuous classification',
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
    config = get_model_config(args.model, task=TASK)

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

    # Initialize classifier
    print(f'\nInitializing EntityTypeClassifier with adapter: {adapter_path}')
    classifier = EntityTypeClassifier(
        model=config, adapter_path=str(adapter_path), device=args.device
    )

    # Load model
    print('Loading model...')
    try:
        classifier.load_model()
    except RuntimeError as e:
        print(f'Error loading model: {e}', file=sys.stderr)
        sys.exit(1)

    # Mode selection: interactive, batch, or single
    if args.interactive:
        # Interactive mode
        run_interactive_mode(classifier)
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
        entity_types = classifier.classify_batch(texts)

        # Format output
        results = []
        for i, (text, entity_type) in enumerate(zip(texts, entity_types), 1):
            results.append(f'Input {i}: {text}')
            results.append(format_entity_type_output(entity_type))
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
        entity_type = classifier.classify(text)

        # Format output
        results = [f'Input: {text}', format_entity_type_output(entity_type)]

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
