import argparse
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.utils.gguf import (
    QUANT_METHOD,
    build_gguf_filename,
    build_repo_stem,
    convert_adapter_to_gguf,
    ensure_private_repo,
    get_hf_username,
    parse_adapter_path,
    resolve_base_model,
    upload_gguf,
    validate_quants,
)


def main():
    """Convert an adapter to GGUF and push it to the Hub."""
    parser = argparse.ArgumentParser(
        description='Convert a trained LoRA adapter to GGUF (default q4_k_m) and '
        'push it to the Hugging Face Hub as a private repo.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run scripts/export/to_gguf.py adapters/address/qwen-0.5b
  uv run scripts/export/to_gguf.py adapters/address/qwen-0.5b --quant q4_k_m q8_0
  uv run scripts/export/to_gguf.py adapters/entity_name/classification/qwen-3b \\
    --repo-id username/name-gguf

The HF token is read from HF_TOKEN (a local .env file is loaded automatically).
Set PUSH_APPROVED=1 to skip the interactive push confirmation.
        """,
    )
    parser.add_argument(
        'adapter_path',
        type=str,
        help='Path to a trained adapter, e.g. adapters/address/qwen-0.5b',
    )
    parser.add_argument(
        '--quant',
        type=str,
        nargs='+',
        default=[QUANT_METHOD],
        metavar='METHOD',
        help=f'One or more quantization methods (default: {QUANT_METHOD}). '
        'Multiple values produce multiple GGUFs in the same repo.',
    )
    parser.add_argument(
        '--repo-id',
        type=str,
        default=None,
        help='Target Hub repo id (default: {username}/{model}_{task})',
    )
    args = parser.parse_args()

    try:
        validate_quants(args.quant)
    except ValueError as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)

    # Load the token from a local .env (does not override an exported HF_TOKEN).
    load_dotenv()
    token = os.environ.get('HF_TOKEN')
    if not token:
        print(
            'Error: HF_TOKEN is not set. Add it to your environment or a .env '
            'file (see .env.example).',
            file=sys.stderr,
        )
        sys.exit(1)

    # Verify HF credentials ("is the account connected?").
    try:
        username = get_hf_username(token)
    except RuntimeError as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)

    # Validate the adapter directory before any heavy work.
    adapter_path = Path(args.adapter_path)
    if not adapter_path.is_dir():
        print(f'Error: adapter path does not exist: {adapter_path}', file=sys.stderr)
        sys.exit(1)
    if not (adapter_path / 'adapter_config.json').is_file():
        print(
            f'Error: no adapter_config.json in {adapter_path}; '
            'is this a trained adapter directory?',
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        task, model_alias = parse_adapter_path(adapter_path)
    except ValueError as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)

    base_model = resolve_base_model(task, model_alias)
    repo_stem = build_repo_stem(base_model, task)
    quant_filenames = {
        quant: build_gguf_filename(base_model, task, quant) for quant in args.quant
    }
    repo_id = args.repo_id or f'{username}/{repo_stem}'

    print('\nGGUF Export Configuration:')
    print(f'  Adapter Path:  {adapter_path}')
    print(f'  Task:          {task}')
    print(f'  Model Alias:   {model_alias}')
    print(f'  Base Model:    {base_model}')
    print(f'  Quantization:  {", ".join(args.quant)}')
    print(f'  GGUF Files:    {", ".join(quant_filenames.values())}')
    print(f'  HF User:       {username}')
    print(f'  Target Repo:   {repo_id} (private)')

    work_dir = Path('outputs/gguf') / repo_stem
    print(f'\nConverting adapter to GGUF in {work_dir} ...')
    print('(First run builds llama.cpp; this can take a while.)')
    gguf_paths = convert_adapter_to_gguf(adapter_path, work_dir, quant_filenames)
    for quant, path in gguf_paths.items():
        print(f'  {quant}: {path}')

    # External-push safety gate.
    if os.environ.get('PUSH_APPROVED') != '1':
        files = ', '.join(path.name for path in gguf_paths.values())
        prompt = f'\nPush {files} to {repo_id} (private)? [y/N] '
        if input(prompt).strip().lower() != 'y':
            print(f'Push cancelled. Local GGUFs kept in: {work_dir}')
            return

    print(f'\nEnsuring private repo {repo_id} ...')
    ensure_private_repo(repo_id, token)

    url = f'https://huggingface.co/{repo_id}'
    for quant, path in gguf_paths.items():
        print(f'Uploading {path.name} ...')
        url = upload_gguf(
            path, repo_id, path.name, token, f'Add {quant} GGUF for {task}'
        )

    readme = adapter_path / 'README.md'
    if readme.is_file():
        print('Uploading README.md ...')
        upload_gguf(readme, repo_id, 'README.md', token, 'Add model card')

    # Clean up the local working directory after a successful push.
    shutil.rmtree(work_dir, ignore_errors=True)

    print('\n' + '=' * 60)
    print('GGUF Export Complete!')
    print('=' * 60)
    print(f'Uploaded {len(gguf_paths)} file(s) to: {url}')
    print('=' * 60)


if __name__ == '__main__':
    main()
