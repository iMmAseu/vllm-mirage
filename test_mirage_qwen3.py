#!/usr/bin/env python3
"""Test script for Mirage-accelerated Qwen3 in vLLM.

This script compares the performance of standard vLLM vs Mirage-accelerated vLLM.
"""

import argparse
import logging
import os
import time
from typing import List

import torch


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(levelname)s] %(name)s: %(message)s'
    )


def run_inference(
    model_name: str,
    prompts: List[str],
    use_mirage: bool,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> tuple:
    """Run inference with or without Mirage.

    Returns:
        tuple: (outputs, latency_ms)
    """
    # Set environment variable
    os.environ['VLLM_USE_MIRAGE'] = '1' if use_mirage else '0'

    # Import vLLM after setting env var
    from vllm import LLM, SamplingParams

    logging.info(f"{'='*80}")
    logging.info(f"Running inference with Mirage: {use_mirage}")
    logging.info(f"{'='*80}")

    # Initialize model
    logging.info("Loading model...")
    start_time = time.time()
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        max_model_len=16,  # Reduced to 16 for 16GB GPU
        trust_remote_code=True,
        enforce_eager=True,  # Disable CUDA graphs to save memory
        gpu_memory_utilization=0.85,  # Reduce memory allocation
    )
    load_time = time.time() - start_time
    logging.info(f"Model loaded in {load_time:.2f}s")

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    # Warmup
    logging.info("Warming up...")
    _ = llm.generate(prompts[:1], sampling_params)

    # Benchmark
    logging.info(f"Generating for {len(prompts)} prompts...")
    torch.cuda.synchronize()
    start_time = time.time()

    outputs = llm.generate(prompts, sampling_params)

    torch.cuda.synchronize()
    end_time = time.time()

    latency_ms = (end_time - start_time) * 1000
    avg_latency_ms = latency_ms / len(prompts)

    logging.info(f"Total latency: {latency_ms:.2f}ms")
    logging.info(f"Average latency per prompt: {avg_latency_ms:.2f}ms")

    return outputs, latency_ms, avg_latency_ms


def main():
    parser = argparse.ArgumentParser(description="Test Mirage-accelerated Qwen3")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model name or path"
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "The quick brown fox",
            "What is the meaning of life?",
            "Explain quantum computing in simple terms:",
            "Write a short story about a robot:",
        ],
        help="Test prompts"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline (non-Mirage) test"
    )
    parser.add_argument(
        "--skip-mirage",
        action="store_true",
        help="Skip Mirage test"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    results = {}

    # Test baseline (standard vLLM)
    if not args.skip_baseline:
        logging.info("\n" + "="*80)
        logging.info("BASELINE TEST (Standard vLLM)")
        logging.info("="*80 + "\n")
        try:
            outputs_baseline, latency_baseline, avg_latency_baseline = run_inference(
                model_name=args.model,
                prompts=args.prompts,
                use_mirage=False,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            results['baseline'] = {
                'outputs': outputs_baseline,
                'latency_ms': latency_baseline,
                'avg_latency_ms': avg_latency_baseline,
            }
        except Exception as e:
            logging.error(f"Baseline test failed: {e}")
            import traceback
            traceback.print_exc()

    # Test Mirage-accelerated
    if not args.skip_mirage:
        logging.info("\n" + "="*80)
        logging.info("MIRAGE TEST (Mirage-Accelerated vLLM)")
        logging.info("="*80 + "\n")
        try:
            outputs_mirage, latency_mirage, avg_latency_mirage = run_inference(
                model_name=args.model,
                prompts=args.prompts,
                use_mirage=True,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            results['mirage'] = {
                'outputs': outputs_mirage,
                'latency_ms': latency_mirage,
                'avg_latency_ms': avg_latency_mirage,
            }
        except Exception as e:
            logging.error(f"Mirage test failed: {e}")
            import traceback
            traceback.print_exc()

    # Print comparison
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80 + "\n")

    if 'baseline' in results:
        print(f"Baseline (Standard vLLM):")
        print(f"  Total latency: {results['baseline']['latency_ms']:.2f}ms")
        print(f"  Avg latency/prompt: {results['baseline']['avg_latency_ms']:.2f}ms")

    if 'mirage' in results:
        print(f"\nMirage (Accelerated vLLM):")
        print(f"  Total latency: {results['mirage']['latency_ms']:.2f}ms")
        print(f"  Avg latency/prompt: {results['mirage']['avg_latency_ms']:.2f}ms")

    if 'baseline' in results and 'mirage' in results:
        speedup = results['baseline']['latency_ms'] / results['mirage']['latency_ms']
        improvement = (1 - results['mirage']['latency_ms'] / results['baseline']['latency_ms']) * 100
        print(f"\nPerformance Comparison:")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Improvement: {improvement:.1f}%")

        if speedup > 1.0:
            print(f"\n✅ Mirage is {speedup:.2f}x faster!")
        elif speedup < 0.95:
            print(f"\n⚠️  Mirage is slower. This may be normal for small batches.")
        else:
            print(f"\n➡️  Performance is similar. Try larger batch sizes.")

    # Print sample outputs
    if 'baseline' in results or 'mirage' in results:
        print("\n" + "="*80)
        print("SAMPLE OUTPUTS")
        print("="*80 + "\n")

        outputs = results.get('mirage', results.get('baseline'))['outputs']
        for i, output in enumerate(outputs[:2]):  # Show first 2
            print(f"Prompt {i+1}: {output.prompt}")
            print(f"Generated: {output.outputs[0].text[:200]}...")
            print("-" * 80)

    print("\n" + "="*80)
    print("Test completed!")
    print("="*80)


if __name__ == "__main__":
    main()
