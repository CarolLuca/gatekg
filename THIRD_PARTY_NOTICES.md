# Third-party notices

GateKG's original code is licensed under the [MIT license](LICENSE).

The answer normalization, token-overlap metrics, supporting-fact metrics and evaluation driver in `src/gatekg/evaluation/hotpot.py` are adapted from HotpotQA. Copyright 2018 Zhilin Yang, Peng Qi, Saizheng Zhang. GateKG packages and formats the evaluator for use by its experiment runners. [Upstream evaluator](https://github.com/hotpotqa/hotpot/blob/3635853403a8735609ee997664e1528f4480762a/hotpot_evaluate_v1.py) · [Apache License 2.0](licenses/HotpotQA.txt)

RLM prompt passages in `src/gatekg/utils/prompts.py` and `src/gatekg/utils/mcp_prompts.py` are adapted from Recursive Language Models, with GateKG-specific graph tools and instructions. Copyright (c) 2026 Alex Zhang. [Upstream RLM](https://github.com/alexzhang13/rlm/tree/e11f1a8f8ee4490686e16dd6463233abe08a07eb) · [MIT license](licenses/RLM.txt)
