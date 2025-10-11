#!/usr/bin/env python3
"""Comprehensive performance benchmark for Mirage-accelerated vLLM.

This script performs a thorough performance comparison between standard vLLM
and Mirage-accelerated vLLM, with proper warmup and statistical analysis.
"""

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(levelname)s] %(message)s'
    )


def load_prompts(prompt_file: str) -> List[str]:
    """Load prompts from file, filtering out comments and empty lines."""
    prompts = []
    with open(prompt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith('#'):
                prompts.append(line)
    return prompts


def run_warmup(llm, prompts: List[str], sampling_params, num_warmup: int = 5):
    """Run warmup iterations to stabilize performance."""
    logging.info(f"Running {num_warmup} warmup iterations...")
    warmup_prompts = prompts[:min(len(prompts), num_warmup)]

    for i in range(num_warmup):
        _ = llm.generate(warmup_prompts, sampling_params)
        torch.cuda.synchronize()

    logging.info("Warmup completed")


def benchmark_batch(
    llm,
    prompts: List[str],
    sampling_params,
    batch_size: int,
    num_iterations: int = 3
) -> Dict[str, any]:
    """Benchmark a batch of prompts multiple times and collect statistics."""

    latencies = []
    tokens_per_second = []
    total_tokens_generated = []

    for iteration in range(num_iterations):
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        outputs = llm.generate(prompts[:batch_size], sampling_params)

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        latency = (end_time - start_time) * 1000  # Convert to ms
        latencies.append(latency)

        # Calculate tokens generated
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        total_tokens_generated.append(total_tokens)

        # Calculate tokens per second
        throughput = total_tokens / (latency / 1000)
        tokens_per_second.append(throughput)

    return {
        'batch_size': batch_size,
        'num_iterations': num_iterations,
        'latency_ms': {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies),
            'median': np.median(latencies),
            'all': latencies,
        },
        'tokens_per_second': {
            'mean': np.mean(tokens_per_second),
            'std': np.std(tokens_per_second),
            'min': np.min(tokens_per_second),
            'max': np.max(tokens_per_second),
            'median': np.median(tokens_per_second),
            'all': tokens_per_second,
        },
        'total_tokens': {
            'mean': np.mean(total_tokens_generated),
            'total': sum(total_tokens_generated),
        }
    }


def run_benchmark(
    model_name: str,
    prompts: List[str],
    use_mirage: bool,
    max_tokens: int = 50,
    temperature: float = 0.8,
    top_p: float = 0.95,
    num_warmup: int = 5,
    num_iterations: int = 5,
    batch_sizes: List[int] = [1, 2, 4],
) -> Dict[str, any]:
    """Run complete benchmark for one configuration."""

    # Set environment variable
    os.environ['VLLM_USE_MIRAGE'] = '1' if use_mirage else '0'

    # Import vLLM after setting env var
    from vllm import LLM, SamplingParams

    config_name = "Mirage" if use_mirage else "Baseline"
    logging.info(f"\n{'='*80}")
    logging.info(f"BENCHMARKING: {config_name}")
    logging.info(f"{'='*80}\n")

    # Initialize model
    logging.info("Loading model...")
    start_time = time.time()
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        max_model_len=128,  # Increased for better benchmarking
        trust_remote_code=True,
        enforce_eager=True,  # Disable CUDA graphs for fair comparison
        gpu_memory_utilization=0.85,
    )
    load_time = time.time() - start_time
    logging.info(f"Model loaded in {load_time:.2f}s")

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    # Run warmup
    run_warmup(llm, prompts, sampling_params, num_warmup)

    # Benchmark different batch sizes
    results = {
        'config': config_name,
        'use_mirage': use_mirage,
        'model_name': model_name,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'num_warmup': num_warmup,
        'num_iterations': num_iterations,
        'load_time_seconds': load_time,
        'batch_results': {}
    }

    for batch_size in batch_sizes:
        if batch_size > len(prompts):
            logging.warning(f"Skipping batch size {batch_size} (only {len(prompts)} prompts available)")
            continue

        logging.info(f"\nBenchmarking batch size: {batch_size}")
        batch_results = benchmark_batch(
            llm, prompts, sampling_params, batch_size, num_iterations
        )
        results['batch_results'][batch_size] = batch_results

        # Print immediate results
        logging.info(f"  Latency: {batch_results['latency_ms']['mean']:.2f} ± {batch_results['latency_ms']['std']:.2f} ms")
        logging.info(f"  Throughput: {batch_results['tokens_per_second']['mean']:.2f} ± {batch_results['tokens_per_second']['std']:.2f} tokens/s")

    return results


def calculate_speedup(baseline_results: Dict, mirage_results: Dict) -> Dict:
    """Calculate speedup metrics comparing Mirage to baseline."""
    speedup_results = {
        'batch_speedups': {}
    }

    for batch_size in baseline_results['batch_results'].keys():
        if batch_size not in mirage_results['batch_results']:
            continue

        baseline = baseline_results['batch_results'][batch_size]
        mirage = mirage_results['batch_results'][batch_size]

        latency_speedup = baseline['latency_ms']['mean'] / mirage['latency_ms']['mean']
        throughput_speedup = mirage['tokens_per_second']['mean'] / baseline['tokens_per_second']['mean']

        speedup_results['batch_speedups'][batch_size] = {
            'latency_speedup': latency_speedup,
            'throughput_speedup': throughput_speedup,
            'latency_reduction_percent': (1 - mirage['latency_ms']['mean'] / baseline['latency_ms']['mean']) * 100,
            'baseline_latency_ms': baseline['latency_ms']['mean'],
            'mirage_latency_ms': mirage['latency_ms']['mean'],
            'baseline_throughput': baseline['tokens_per_second']['mean'],
            'mirage_throughput': mirage['tokens_per_second']['mean'],
        }

    return speedup_results


def generate_report(
    baseline_results: Dict,
    mirage_results: Dict,
    speedup_results: Dict,
    output_file: str = None
):
    """Generate a comprehensive performance report."""

    report = []
    report.append("\n" + "="*100)
    report.append("MIRAGE PERFORMANCE BENCHMARK REPORT")
    report.append("="*100 + "\n")

    # Timestamp
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Configuration
    report.append("CONFIGURATION")
    report.append("-" * 100)
    report.append(f"Model: {baseline_results['model_name']}")
    report.append(f"Max tokens: {baseline_results['max_tokens']}")
    report.append(f"Temperature: {baseline_results['temperature']}")
    report.append(f"Top-p: {baseline_results['top_p']}")
    report.append(f"Warmup iterations: {baseline_results['num_warmup']}")
    report.append(f"Benchmark iterations: {baseline_results['num_iterations']}")
    report.append(f"Model load time - Baseline: {baseline_results['load_time_seconds']:.2f}s")
    report.append(f"Model load time - Mirage: {mirage_results['load_time_seconds']:.2f}s\n")

    # Detailed results by batch size
    report.append("DETAILED RESULTS BY BATCH SIZE")
    report.append("-" * 100)

    for batch_size in sorted(baseline_results['batch_results'].keys()):
        if batch_size not in mirage_results['batch_results']:
            continue

        baseline = baseline_results['batch_results'][batch_size]
        mirage = mirage_results['batch_results'][batch_size]
        speedup = speedup_results['batch_speedups'][batch_size]

        report.append(f"\nBatch Size: {batch_size}")
        report.append("  " + "-" * 96)

        # Latency comparison
        report.append(f"  LATENCY (ms):")
        report.append(f"    Baseline:  {baseline['latency_ms']['mean']:8.2f} ± {baseline['latency_ms']['std']:6.2f}  (min: {baseline['latency_ms']['min']:8.2f}, max: {baseline['latency_ms']['max']:8.2f}, median: {baseline['latency_ms']['median']:8.2f})")
        report.append(f"    Mirage:    {mirage['latency_ms']['mean']:8.2f} ± {mirage['latency_ms']['std']:6.2f}  (min: {mirage['latency_ms']['min']:8.2f}, max: {mirage['latency_ms']['max']:8.2f}, median: {mirage['latency_ms']['median']:8.2f})")
        report.append(f"    Speedup:   {speedup['latency_speedup']:.3f}x  ({speedup['latency_reduction_percent']:+.1f}% latency reduction)")

        # Throughput comparison
        report.append(f"  THROUGHPUT (tokens/s):")
        report.append(f"    Baseline:  {baseline['tokens_per_second']['mean']:8.2f} ± {baseline['tokens_per_second']['std']:6.2f}")
        report.append(f"    Mirage:    {mirage['tokens_per_second']['mean']:8.2f} ± {mirage['tokens_per_second']['std']:6.2f}")
        report.append(f"    Speedup:   {speedup['throughput_speedup']:.3f}x")

    # Summary
    report.append("\n" + "="*100)
    report.append("SUMMARY")
    report.append("="*100)

    avg_latency_speedup = np.mean([s['latency_speedup'] for s in speedup_results['batch_speedups'].values()])
    avg_throughput_speedup = np.mean([s['throughput_speedup'] for s in speedup_results['batch_speedups'].values()])

    report.append(f"\nAverage Latency Speedup:    {avg_latency_speedup:.3f}x")
    report.append(f"Average Throughput Speedup: {avg_throughput_speedup:.3f}x")

    if avg_latency_speedup > 1.1:
        report.append(f"\n✅ Mirage is {avg_latency_speedup:.2f}x faster than baseline!")
    elif avg_latency_speedup > 0.95:
        report.append(f"\n➡️  Performance is similar (within 5%). Speedup: {avg_latency_speedup:.2f}x")
    else:
        report.append(f"\n⚠️  Mirage is slower. This may be due to overhead or small batch sizes. Speedup: {avg_latency_speedup:.2f}x")

    report.append("\n" + "="*100 + "\n")

    # Print report
    report_text = "\n".join(report)
    print(report_text)

    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        logging.info(f"\nReport saved to: {output_file}")

    return report_text


def save_json_results(baseline_results: Dict, mirage_results: Dict, speedup_results: Dict, output_file: str):
    """Save detailed results to JSON file."""
    results = {
        'timestamp': datetime.now().isoformat(),
        'baseline': baseline_results,
        'mirage': mirage_results,
        'speedup': speedup_results,
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logging.info(f"Detailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Mirage-accelerated vLLM")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model name or path"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="test_prompts.txt",
        help="Path to prompts file"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
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
        "--num-warmup",
        type=int,
        default=5,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=5,
        help="Number of benchmark iterations per batch size"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Batch sizes to test"
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline test"
    )
    parser.add_argument(
        "--skip-mirage",
        action="store_true",
        help="Skip Mirage test"
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="benchmark_report.txt",
        help="Output report file"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="benchmark_results.json",
        help="Output JSON results file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    # Load prompts
    logging.info(f"Loading prompts from: {args.prompts}")
    prompts = load_prompts(args.prompts)
    logging.info(f"Loaded {len(prompts)} prompts\n")

    baseline_results = None
    mirage_results = None

    # Run baseline benchmark
    if not args.skip_baseline:
        baseline_results = run_benchmark(
            model_name=args.model,
            prompts=prompts,
            use_mirage=False,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_warmup=args.num_warmup,
            num_iterations=args.num_iterations,
            batch_sizes=args.batch_sizes,
        )

    # Run Mirage benchmark
    if not args.skip_mirage:
        mirage_results = run_benchmark(
            model_name=args.model,
            prompts=prompts,
            use_mirage=True,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_warmup=args.num_warmup,
            num_iterations=args.num_iterations,
            batch_sizes=args.batch_sizes,
        )

    # Generate report if we have both results
    if baseline_results and mirage_results:
        speedup_results = calculate_speedup(baseline_results, mirage_results)
        generate_report(baseline_results, mirage_results, speedup_results, args.output_report)
        save_json_results(baseline_results, mirage_results, speedup_results, args.output_json)
    elif baseline_results:
        logging.info("\nOnly baseline results available (Mirage test was skipped)")
    elif mirage_results:
        logging.info("\nOnly Mirage results available (baseline test was skipped)")


if __name__ == "__main__":
    main()
