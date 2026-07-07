"""Score the vLLM-served entity-type classification adapter on a dataset split.

The served ``classification`` adapter is the ``classification_external``
variant, evaluated against the matching external validation split.
"""

from src.vllm.evaluation import main
from src.vllm.tasks import CLASSIFICATION

if __name__ == '__main__':
    main(CLASSIFICATION)
