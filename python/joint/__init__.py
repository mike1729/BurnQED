"""v2 joint LLM+EBM training pipeline.

Implements joint LoRA + GoalConditioned EBM training with shared gradients
through the backbone. Based on CURL (Laskin 2020) and VLM joint training.
"""
