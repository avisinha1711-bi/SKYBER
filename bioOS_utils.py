"""
BioOS Utility Functions
Helper functions and utilities for biological operating system
"""

import logging
import json
import random
from typing import List, Dict, Any, Tuple
from datetime import datetime
from enum import Enum
import bioOS_config as config

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_file: str = config.LOG_FILE, 
                  log_level: str = config.LOG_LEVEL) -> logging.Logger:
    """
    Configure logging for BioOS
    
    Args:
        log_file: Path to log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("BioOS")
    logger.setLevel(getattr(logging, log_level))
    
    # Console handler
    if config.LOG_TO_CONSOLE:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level))
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if config.LOG_TO_FILE:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level))
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()

# ============================================================================
# GENETIC SEQUENCE UTILITIES
# ============================================================================

def generate_random_sequence(length: int = 12) -> str:
    """
    Generate random DNA sequence
    
    Args:
        length: Length of sequence
    
    Returns:
        Random DNA sequence string (A, T, G, C)
    """
    nucleotides = ['A', 'T', 'G', 'C']
    return ''.join(random.choice(nucleotides) for _ in range(length))

def reverse_complement(sequence: str) -> str:
    """
    Get reverse complement of DNA sequence
    
    Args:
        sequence: DNA sequence
    
    Returns:
        Reverse complement sequence
    """
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    return ''.join(complement.get(base, 'N') for base in reversed(sequence))

def hamming_distance(seq1: str, seq2: str) -> int:
    """
    Calculate Hamming distance between two sequences
    
    Args:
        seq1: First sequence
        seq2: Second sequence
    
    Returns:
        Number of differing positions
    """
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be same length")
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))

def mutate_sequence(sequence: str, mutation_rate: float = 0.01) -> str:
    """
    Introduce mutations in DNA sequence
    
    Args:
        sequence: Original sequence
        mutation_rate: Probability of mutation per base
    
    Returns:
        Mutated sequence
    """
    nucleotides = ['A', 'T', 'G', 'C']
    mutated = list(sequence)
    
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] = random.choice(nucleotides)
    
    return ''.join(mutated)

def translate_codon(codon: str) -> str:
    """
    Translate DNA codon to amino acid (simplified)
    
    Args:
        codon: 3-base codon
    
    Returns:
        Amino acid code
    """
    codon_table = {
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
        'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
        'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
        'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
        'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
    }
    return codon_table.get(codon.upper(), 'X')

# ============================================================================
# STATISTICAL UTILITIES
# ============================================================================

def calculate_stats(values: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistics on list of values
    
    Args:
        values: List of numeric values
    
    Returns:
        Dictionary with min, max, mean, median, stddev
    """
    if not values:
        return {}
    
    sorted_vals = sorted(values)
    n = len(values)
    mean = sum(values) / n
    median = sorted_vals[n // 2]
    variance = sum((x - mean) ** 2 for x in values) / n
    stddev = variance ** 0.5
    
    return {
        'min': min(values),
        'max': max(values),
        'mean': mean,
        'median': median,
        'stddev': stddev,
        'count': n
    }

def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize value to range [0, 1]
    
    Args:
        value: Value to normalize
        min_val: Minimum of range
        max_val: Maximum of range
    
    Returns:
        Normalized value
    """
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)

def weighted_random_choice(choices: Dict[Any, float]) -> Any:
    """
    Select random choice weighted by probabilities
    
    Args:
        choices: Dictionary of {choice: weight}
    
    Returns:
        Selected choice
    """
    total = sum(choices.values())
    pick = random.uniform(0, total)
    current = 0
    
    for choice, weight in choices.items():
        current += weight
        if pick <= current:
            return choice
    
    return list(choices.keys())[-1]

# ============================================================================
# PROCESS UTILITIES
# ============================================================================

def format_process_info(process: Any) -> str:
    """
    Format process information for display
    
    Args:
        process: BioProcess instance
    
    Returns:
        Formatted string representation
    """
    return (
        f"PID: {process.pid} | "
        f"Name: {process.name} | "
        f"State: {process.state.value} | "
        f"Energy: {process.energy:.2f} | "
        f"Age: {process.age:.2f}"
    )

def estimate_population_fitness(processes: List[Any]) -> Dict[str, float]:
    """
    Calculate fitness metrics for population
    
    Args:
        processes: List of BioProcess instances
    
    Returns:
        Dictionary with fitness statistics
    """
    if not processes:
        return {}
    
    energies = [p.energy for p in processes]
    ages = [p.age for p in processes]
    
    return {
        'avg_energy': sum(energies) / len(energies),
        'avg_age': sum(ages) / len(ages),
        'population_size': len(processes),
        'max_energy': max(energies),
        'max_age': max(ages)
    }

# ============================================================================
# FILE I/O UTILITIES
# ============================================================================

def save_simulation_state(filename: str, state: Dict[str, Any]) -> bool:
    """
    Save simulation state to JSON file
    
    Args:
        filename: Output filename
        state: Simulation state dictionary
    
    Returns:
        Success status
    """
    try:
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        logger.info(f"Simulation state saved to {filename}")
        return True
    except Exception as e:
        logger.error(f"Failed to save state: {e}")
        return False

def load_simulation_state(filename: str) -> Dict[str, Any]:
    """
    Load simulation state from JSON file
    
    Args:
        filename: Input filename
    
    Returns:
        Simulation state dictionary
    """
    try:
        with open(filename, 'r') as f:
            state = json.load(f)
        logger.info(f"Simulation state loaded from {filename}")
        return state
    except Exception as e:
        logger.error(f"Failed to load state: {e}")
        return {}

# ============================================================================
# PERFORMANCE UTILITIES
# ============================================================================

def profile_function(func):
    """
    Decorator to profile function execution time
    
    Args:
        func: Function to profile
    
    Returns:
        Wrapped function with profiling
    """
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.debug(f"{func.__name__} took {elapsed:.4f}s")
        return result
    
    return wrapper

def get_memory_usage() -> float:
    """
    Get current memory usage in MB
    
    Returns:
        Memory usage in megabytes
    """
    import os
    import psutil
    
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_dna_sequence(sequence: str) -> bool:
    """
    Validate DNA sequence contains only ATGC
    
    Args:
        sequence: DNA sequence string
    
    Returns:
        Validation result
    """
    return all(base in 'ATGC' for base in sequence.upper())

def validate_process_state(process: Any) -> bool:
    """
    Validate process integrity
    
    Args:
        process: BioProcess instance
    
    Returns:
        Validation result
    """
    return (
        process.pid >= 0 and
        process.name and
        process.energy >= 0 and
        process.age >= 0
    )

# ============================================================================
# REPORTING UTILITIES
# ============================================================================

def generate_simulation_report(timestamp: float, 
                               processes: List[Any],
                               memory_usage: float) -> str:
    """
    Generate text report of simulation status
    
    Args:
        timestamp: Current simulation time
        processes: List of active processes
        memory_usage: Memory usage percentage
    
    Returns:
        Formatted report string
    """
    fitness = estimate_population_fitness(processes)
    
    report = f"""
{'='*60}
BIOÖS SIMULATION REPORT
{'='*60}
Timestamp: {timestamp:.2f}s
Active Processes: {len(processes)}
Memory Usage: {memory_usage:.2f}%

Population Statistics:
  Average Energy: {fitness.get('avg_energy', 0):.2f}
  Average Age: {fitness.get('avg_age', 0):.2f}
  Max Energy: {fitness.get('max_energy', 0):.2f}
  Max Age: {fitness.get('max_age', 0):.2f}
{'='*60}
"""
    return report
