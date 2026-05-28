"""Inference script for spaCy-based entity name extraction."""

import argparse
import sys
from pathlib import Path

from src.entity_name.inference import EntityNameExtractionPipeline
from src.entity_name.models import EntityName
from src.utils import save_file


def format_entity_output(name: EntityName, input_text: str) -> str:
    """Format an extraction result for display.

    Args:
        name: ``EntityName`` produced by the pipeline.
        input_text: The raw input that produced this result.

    Returns:
        Multi-line formatted result.
    """
    lines = [
        'Result:',
        f'  Input:  {input_text}',
        f'  Entity: {name.to_output()}',
    ]
    return '\n'.join(lines)


def run_interactive_mode(pipeline: EntityNameExtractionPipeline) -> None:
    """Run an interactive REPL for entity-name extraction.

    Args:
        pipeline: Initialized ``EntityNameExtractionPipeline`` with a loaded
            spaCy model.
    """
    print('\n' + '=' * 60)
    print('Interactive Entity Name Extraction Mode')
    print('=' * 60)
    print('\nAvailable commands:')
    print('  exit, quit, q    - Exit the session')
    print('  help, ?          - Display this help message')
    print('\nEnter text containing an entity name to extract, or a command.')
    print('=' * 60 + '\n')

    while True:
        try:
            user_input = input('> ').strip()
            if not user_input:
                continue

            command = user_input.lower()
            if command in ('exit', 'quit', 'q'):
                print('\nGoodbye!')
                break
            if command in ('help', '?'):
                print('\nAvailable commands:')
                print('  exit, quit, q    - Exit')
                print('  help, ?          - Help')
                continue

            try:
                name = pipeline.predict(user_input)
                print()
                print(format_entity_output(name, user_input))
                print()
            except RuntimeError as e:
                print(f'\nError: {e}')

        except KeyboardInterrupt:
            print('\n\nInterrupted. Exiting...')
            break
        except EOFError:
            print('\n\nGoodbye!')
            break


def main():
    """Run inference pipeline."""
    parser = argparse.ArgumentParser(
        description='Extract entity names from text using a spaCy NER pipeline'
    )
    parser.add_argument(
        'input',
        type=str,
        nargs='?',
        help='Text input or path to file (one record per line)',
    )
    parser.add_argument(
        '--spacy-model',
        type=str,
        default='en_core_web_trf',
        help='spaCy pipeline name (default: en_core_web_trf)',
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Treat input as file path and process lines in batch',
    )
    parser.add_argument(
        '--interactive',
        '-i',
        action='store_true',
        help='Launch interactive REPL',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: print to stdout)',
    )
    args = parser.parse_args()

    if args.interactive and args.batch:
        print(
            'Error: --interactive and --batch are mutually exclusive',
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.interactive and not args.input:
        print(
            'Error: input argument is required for non-interactive mode',
            file=sys.stderr,
        )
        sys.exit(1)

    print('\nPipeline Configuration:')
    print(f'  spaCy model:      {args.spacy_model}')

    pipeline = EntityNameExtractionPipeline(spacy_model=args.spacy_model)

    print('Loading spaCy model...')
    try:
        pipeline.load_model()
    except RuntimeError as e:
        print(f'Error loading model: {e}', file=sys.stderr)
        sys.exit(1)

    if args.interactive:
        run_interactive_mode(pipeline)
        return

    if args.batch:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f'Error: input file does not exist: {input_path}', file=sys.stderr)
            sys.exit(1)

        with open(input_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

        if not texts:
            print('Error: No valid input lines found in file', file=sys.stderr)
            sys.exit(1)

        print(f'Processing {len(texts)} inputs in batch...\n')
        names = pipeline.predict_batch(texts)

        results = []
        for i, (text, name) in enumerate(zip(texts, names), 1):
            results.append(f'Input {i}:')
            results.append(format_entity_output(name, text))
            results.append('')
    else:
        input_path = Path(args.input)
        if input_path.exists():
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        else:
            text = args.input

        if not text:
            print('Error: Empty input provided', file=sys.stderr)
            sys.exit(1)

        name = pipeline.predict(text)
        results = [format_entity_output(name, text)]

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
