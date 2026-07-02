import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from src.pipeline.config import (
    CLASSIFICATION_TASKS,
    DEFAULT_ADDRESS_MODEL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLASSIFICATION_MODEL,
    DEFAULT_CLASSIFICATION_TASK,
    DEFAULT_EXTRACTION_MODEL,
    resolve_stage_configs,
)
from src.pipeline.models import PipelineResult
from src.utils import save_file

if TYPE_CHECKING:
    from src.pipeline.inference import ExtractionPipeline


def format_result_json(result: PipelineResult) -> str:
    """Format a single pipeline result as pretty-printed JSON.

    Args:
        result: PipelineResult to format

    Returns:
        Pretty-printed JSON string
    """
    return json.dumps(result.model_dump(), indent=2)


def format_results_jsonl(results: list[PipelineResult]) -> str:
    """Format pipeline results as JSONL, one compact object per line.

    Args:
        results: PipelineResult objects in input order

    Returns:
        JSONL string with one JSON object per line
    """
    return '\n'.join(json.dumps(result.model_dump()) for result in results)


def run_interactive_mode(pipeline: 'ExtractionPipeline') -> None:
    """Run an interactive session for the combined pipeline.

    Prompts and status go to stderr; result JSON goes to stdout.

    Args:
        pipeline: Initialized ExtractionPipeline with loaded models
    """
    # Display welcome message
    print('\n' + '=' * 60, file=sys.stderr)
    print('Interactive Combined Extraction Pipeline Mode', file=sys.stderr)
    print('=' * 60, file=sys.stderr)
    print('\nAvailable commands:', file=sys.stderr)
    print('  exit, quit, q    - Exit the session', file=sys.stderr)
    print('  help, ?          - Display this help message', file=sys.stderr)
    print('\nEnter a raw combined line to process, or a command.', file=sys.stderr)
    print('=' * 60 + '\n', file=sys.stderr)

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
                print('\nGoodbye!', file=sys.stderr)
                running = False
                continue

            # Handle help commands
            if command in ['help', '?']:
                print('\nAvailable commands:', file=sys.stderr)
                print('  exit, quit, q    - Exit the session', file=sys.stderr)
                print(
                    '  help, ?          - Display this help message',
                    file=sys.stderr,
                )
                print('\nEnter a raw combined line to process.\n', file=sys.stderr)
                continue

            # Run the pipeline on the input line
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    result = pipeline.run(user_input)

                # Display the result JSON on stdout
                print()
                print(format_result_json(result))
                print()

            except RuntimeError as e:
                print(f'\nError: {e}', file=sys.stderr)
                print('Please try again.\n', file=sys.stderr)

        except KeyboardInterrupt:
            print('\n\nInterrupted. Exiting...', file=sys.stderr)
            break
        except EOFError:
            print('\n\nGoodbye!', file=sys.stderr)
            break


def main():
    """Run the combined pipeline CLI."""
    parser = argparse.ArgumentParser(
        description=(
            'Run the combined address + entity name + entity type pipeline '
            'on raw combined lines'
        )
    )
    parser.add_argument(
        'input',
        type=str,
        nargs='?',
        help=(
            'Raw combined line, or path to a file (with --batch: one input per line)'
        ),
    )
    parser.add_argument(
        '--address-model',
        type=str,
        default=DEFAULT_ADDRESS_MODEL,
        help=f'Model alias for the address stage (default: {DEFAULT_ADDRESS_MODEL})',
    )
    parser.add_argument(
        '--extraction-model',
        type=str,
        default=DEFAULT_EXTRACTION_MODEL,
        help=(
            'Model alias for the name extraction stage '
            f'(default: {DEFAULT_EXTRACTION_MODEL})'
        ),
    )
    parser.add_argument(
        '--classification-model',
        type=str,
        default=DEFAULT_CLASSIFICATION_MODEL,
        help=(
            'Model alias for the classification stage '
            f'(default: {DEFAULT_CLASSIFICATION_MODEL})'
        ),
    )
    parser.add_argument(
        '--classification-task',
        type=str,
        default=DEFAULT_CLASSIFICATION_TASK,
        choices=sorted(CLASSIFICATION_TASKS),
        help=(
            'Classification task variant whose adapters to use '
            f'(default: {DEFAULT_CLASSIFICATION_TASK})'
        ),
    )
    parser.add_argument(
        '--concurrent',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Load stage models concurrently (default: false)',
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Treat input as file path and process lines in batch (JSONL output)',
    )
    parser.add_argument(
        '--interactive',
        '-i',
        action='store_true',
        help='Launch interactive mode for continuous processing',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: print to stdout)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f'Chunk size for --batch processing (default: {DEFAULT_BATCH_SIZE})',
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

    if args.batch_size < 1:
        print('Error: --batch-size must be >= 1', file=sys.stderr)
        sys.exit(1)

    # Resolve per-stage model configurations
    try:
        configs = resolve_stage_configs(
            address_model=args.address_model,
            extraction_model=args.extraction_model,
            classification_model=args.classification_model,
            classification_task=args.classification_task,
        )
    except ValueError as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)

    # Resolve adapter paths from config and fail fast on missing ones
    # before the expensive model imports
    adapter_paths = {
        stage: Path(config.adapter_dir) for stage, config in configs.items()
    }

    missing = [path for path in adapter_paths.values() if not path.exists()]
    if missing:
        for path in missing:
            print(f'Error: Adapter path does not exist: {path}', file=sys.stderr)
        sys.exit(1)

    # Batch mode: read the input file before loading any model
    texts: list[str] = []
    if args.batch:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f'Error: Input file does not exist: {input_path}', file=sys.stderr)
            sys.exit(1)

        with open(input_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

        if not texts:
            print('Error: No valid input lines found in file', file=sys.stderr)
            sys.exit(1)

    # Display pipeline configuration
    stage_aliases = {
        'address': args.address_model,
        'extraction': args.extraction_model,
        'classification': args.classification_model,
    }
    print('\nPipeline Configuration:', file=sys.stderr)
    print(f'  Classification Task: {args.classification_task}', file=sys.stderr)
    for stage, config in configs.items():
        print(
            f'  {stage}: alias={stage_aliases[stage]} '
            f'base={config.base_model} adapter={adapter_paths[stage]}',
            file=sys.stderr,
        )

    # Import and load the three models; the redirect keeps unsloth import
    # banners and stage load messages off stdout
    print('\nLoading pipeline models...', file=sys.stderr)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from src.pipeline.inference import ExtractionPipeline

            pipeline = ExtractionPipeline(
                address_model=args.address_model,
                extraction_model=args.extraction_model,
                classification_model=args.classification_model,
                classification_task=args.classification_task,
            )
            pipeline.load_models(concurrent=args.concurrent)
    except RuntimeError as e:
        print(f'Error loading models: {e}', file=sys.stderr)
        sys.exit(1)

    # Mode selection: interactive, batch, or single
    if args.interactive:
        run_interactive_mode(pipeline)
        return

    if args.batch:
        print(f'Processing {len(texts)} inputs in batch...', file=sys.stderr)
        try:
            with contextlib.redirect_stdout(sys.stderr):
                results = pipeline.run_batch(texts, batch_size=args.batch_size)
        except RuntimeError as e:
            print(f'Error during batch inference: {e}', file=sys.stderr)
            sys.exit(1)

        output_text = format_results_jsonl(results)

    else:
        # Single mode: treat input as text or file
        input_path = Path(args.input)

        if input_path.exists():
            print(f'Reading input from {input_path}...', file=sys.stderr)
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        else:
            text = args.input

        if not text:
            print('Error: Empty input provided', file=sys.stderr)
            sys.exit(1)

        print(f'Processing input: {text}', file=sys.stderr)
        try:
            with contextlib.redirect_stdout(sys.stderr):
                result = pipeline.run(text)
        except RuntimeError as e:
            print(f'Error during inference: {e}', file=sys.stderr)
            sys.exit(1)

        output_text = format_result_json(result)

    # Emit results: file via save_file, else the only stdout write
    if args.output:
        save_file(output_text + '\n', args.output)
        print(f'\nResults saved to {args.output}', file=sys.stderr)
    else:
        print(output_text)


if __name__ == '__main__':
    main()
