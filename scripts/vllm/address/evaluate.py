"""Score the vLLM-served address adapter on a dataset split."""

from src.vllm.evaluation import main
from src.vllm.tasks import ADDRESS

if __name__ == '__main__':
    main(ADDRESS)
