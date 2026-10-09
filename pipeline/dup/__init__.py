# -*- coding: utf-8 -*-
"""He thong Duplicate / Similar Bug Retrieval cho du lieu Jira QA.

Pipeline E2E, moi phase la mot module rieng va bat/tat duoc qua DupConfig:

    Preprocessor -> StructuredExtractor -> CandidateGenerator
                 -> PairFeatureExtractor -> HybridScorer
                 -> Reranker -> DecisionEngine

Xem pipeline/dup/pipeline.py cho lop dieu phoi (DuplicatePipeline).
"""
from .config import DupConfig, load_config          # noqa: F401
from .pipeline import DuplicatePipeline             # noqa: F401
