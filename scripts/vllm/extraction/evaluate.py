"""Score the vLLM-served entity-name extraction adapter on a dataset split."""

from src.vllm.evaluation import main
from src.vllm.tasks import EXTRACTION

if __name__ == '__main__':
    main(EXTRACTION)
