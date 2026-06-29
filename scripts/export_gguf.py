"""Merge a trained LoRA adapter into its base and export a self-contained GGUF.

Option 2 (self-contained): loads the base referenced by the adapter, bakes in the
LoRA weights, and writes a single quantized GGUF that needs nothing else to serve.
The GGUF is then uploaded to a private HuggingFace repo derived from the adapter path.

Usage:
    uv run scripts/export_gguf.py \
        adapters/address/qwen-0.5b \
        --output-dir exports/address-qwen05b \
        --quant q4_k_m

HF_TOKEN must be set in the environment or in a local .env file.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

from unsloth import FastLanguageModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('adapter_path', type=str, help='path to the LoRA adapter dir')
    parser.add_argument(
        '--output-dir', type=str, required=True, help='dir to write the GGUF into'
    )
    parser.add_argument(
        '--quant', type=str, default='q4_k_m', help='GGUF quantization method'
    )
    parser.add_argument('--max-seq-length', type=int, default=2048)
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path).resolve()
    output_dir = Path(args.output_dir).resolve()

    load_dotenv()
    token = os.environ.get('HF_TOKEN')
    if not token:
        print('Error: HF_TOKEN is not set. Add it to your environment or a .env file.', file=sys.stderr)
        sys.exit(1)

    try:
        username = HfApi().whoami(token=token)['name']
    except Exception as exc:
        print(f'Error: HuggingFace authentication failed: {exc}', file=sys.stderr)
        sys.exit(1)

    rel = adapter_path.as_posix().split('adapters/')[-1]  # e.g. 'address/qwen-0.5b'
    repo_id = f"{username}/{rel.replace('/', '-')}-gguf"  # e.g. 'user/address-qwen-0.5b-gguf'

    print(f'Target HF repo: {repo_id} (private)')

    # Unsloth detects adapter_config.json and loads base + adapter together.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    # Merges LoRA into the (dequantized) base and converts/quantizes to GGUF.
    model.save_pretrained_gguf(
        str(output_dir), tokenizer, quantization_method=args.quant
    )

    # Unsloth sometimes writes to a sibling dir with a '_gguf' suffix.
    gguf_dir = output_dir
    alt_dir = output_dir.parent / (output_dir.name + '_gguf')
    if alt_dir.is_dir():
        gguf_dir = alt_dir

    gguf_files = sorted(gguf_dir.glob('*.gguf'))
    if not gguf_files:
        print(f'Error: no GGUF files found in {gguf_dir}', file=sys.stderr)
        sys.exit(1)

    print(f'Wrote GGUF ({args.quant}) to: {gguf_dir}')

    create_repo(repo_id, repo_type='model', private=True, exist_ok=True, token=token)

    api = HfApi()
    for gguf in gguf_files:
        print(f'Uploading {gguf.name} ...')
        api.upload_file(
            path_or_fileobj=str(gguf),
            path_in_repo=gguf.name,
            repo_id=repo_id,
            repo_type='model',
            token=token,
            commit_message=f'Add {args.quant} GGUF',
        )

    readme = adapter_path / 'README.md'
    if readme.is_file():
        print('Uploading README.md ...')
        api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo='README.md',
            repo_id=repo_id,
            repo_type='model',
            token=token,
            commit_message='Add model card',
        )

    print(f'Uploaded to: https://huggingface.co/{repo_id}')


if __name__ == '__main__':
    main()
