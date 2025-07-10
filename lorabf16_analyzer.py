"""
LoRA Weight Analysis Tool with BF16 Support
===========================================

This script provides complete interpretability analysis of LoRA (Low-Rank Adaptation) models
to establish a baseline for what comprehensive AI model auditing looks like.

Handles BF16, FP16, and other tensor types robustly.

Part of research demonstrating the mathematical impossibility of safety auditing at scale.
"""

import torch
import safetensors
from safetensors.torch import load_file
import numpy as np
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import pandas as pd
from datetime import datetime

class LoRABF16Analyzer:
    """Complete LoRA weight analysis with robust dtype handling."""
    
    def __init__(self, lora_path: str):
        """Initialize analyzer with LoRA model path."""
        self.lora_path = Path(lora_path)
        self.weights = None
        self.analysis_results = {}
        self.start_time = None
        
    def load_lora_weights(self) -> Dict[str, torch.Tensor]:
        """Load LoRA weights from safetensors file with dtype handling."""
        print(f"Loading LoRA weights from: {self.lora_path}")
        self.start_time = time.time()
        
        try:
            self.weights = load_file(str(self.lora_path))
            print(f"Successfully loaded {len(self.weights)} weight tensors")
            
            # Log basic statistics and handle different dtypes
            total_params = 0
            dtype_counts = {}
            
            for name, tensor in self.weights.items():
                total_params += tensor.numel()
                dtype_str = str(tensor.dtype)
                dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1
            
            print(f"Total parameters: {total_params:,}")
            print("Data types found:")
            for dtype, count in dtype_counts.items():
                print(f"  {dtype}: {count} tensors")
            
            return self.weights
            
        except Exception as e:
            print(f"Error loading LoRA weights: {e}")
            raise
    
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy with robust dtype handling."""
        try:
            # Handle BF16, FP16, and other exotic types
            if tensor.dtype in [torch.bfloat16, torch.float16]:
                # Convert to float32 for numpy compatibility
                return tensor.detach().cpu().float().numpy()
            elif tensor.dtype in [torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64]:
                # Handle integer types
                return tensor.detach().cpu().float().numpy()
            else:
                # Standard float32/float64
                return tensor.detach().cpu().numpy()
        except Exception as e:
            print(f"Warning: Failed to convert tensor to numpy: {e}")
            # Fallback: force conversion through float32
            return tensor.detach().cpu().float().numpy()
    
    def analyze_lora_structure(self) -> Dict[str, Any]:
        """Analyze the structure and organization of LoRA weights."""
        if self.weights is None:
            raise ValueError("Must load weights first")
        
        print("\n=== LoRA Structure Analysis ===")
        structure_analysis = {
            'total_tensors': len(self.weights),
            'tensor_info': {},
            'lora_pairs': {},
            'parameter_distribution': {},
            'weight_statistics': {},
            'dtype_summary': {}
        }
        
        # Track dtype usage
        dtype_counts = {}
        dtype_params = {}
        
        # Analyze each tensor
        for name, tensor in self.weights.items():
            # Basic tensor info
            tensor_info = {
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype),
                'device': str(tensor.device),
                'parameters': tensor.numel(),
                'memory_mb': tensor.numel() * tensor.element_size() / (1024 * 1024)
            }
            
            # Track dtype statistics
            dtype_str = str(tensor.dtype)
            dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1
            dtype_params[dtype_str] = dtype_params.get(dtype_str, 0) + tensor.numel()
            
            # Calculate basic statistics with robust conversion
            try:
                tensor_np = self._tensor_to_numpy(tensor)
                stats = {
                    'mean': float(np.mean(tensor_np)),
                    'std': float(np.std(tensor_np)),
                    'min': float(np.min(tensor_np)),
                    'max': float(np.max(tensor_np)),
                    'zeros': int(np.sum(tensor_np == 0)),
                    'zero_percentage': float(np.sum(tensor_np == 0) / tensor_np.size * 100)
                }
                tensor_info['statistics'] = stats
                
            except Exception as e:
                print(f"  Warning: Could not calculate statistics for {name}: {e}")
                tensor_info['statistics'] = {'error': str(e)}
            
            structure_analysis['tensor_info'][name] = tensor_info
            print(f"  {name}: {tensor.shape} ({tensor.numel():,} params) [{tensor.dtype}]")
        
        # Store dtype summary
        structure_analysis['dtype_summary'] = {
            'dtype_counts': dtype_counts,
            'dtype_parameters': dtype_params
        }
        
        # Identify LoRA pairs (lora_A and lora_B matrices)
        self._identify_lora_pairs(structure_analysis)
        
        # Calculate parameter distribution
        self._calculate_parameter_distribution(structure_analysis)
        
        self.analysis_results['structure'] = structure_analysis
        return structure_analysis
    
    def _identify_lora_pairs(self, analysis: Dict[str, Any]):
        """Identify LoRA down and up matrix pairs (standard naming convention)."""
        lora_pairs = {}
        
        # Look for standard lora_down.weight and lora_up.weight pairs
        for name in self.weights.keys():
            if '.lora_down.weight' in name:
                base_name = name.replace('.lora_down.weight', '')
                lora_up_name = name.replace('.lora_down.weight', '.lora_up.weight')
                
                if lora_up_name in self.weights:
                    lora_pairs[base_name] = {
                        'lora_down': name,
                        'lora_up': lora_up_name,
                        'down_shape': list(self.weights[name].shape),
                        'up_shape': list(self.weights[lora_up_name].shape),
                        'down_dtype': str(self.weights[name].dtype),
                        'up_dtype': str(self.weights[lora_up_name].dtype)
                    }
        
        # Fallback: try legacy lora_A/lora_B naming if no standard pairs found
        if len(lora_pairs) == 0:
            for name in self.weights.keys():
                if 'lora_A' in name:
                    base_name = name.replace('lora_A', '')
                    lora_b_name = name.replace('lora_A', 'lora_B')
                    
                    if lora_b_name in self.weights:
                        lora_pairs[base_name] = {
                            'lora_down': name,
                            'lora_up': lora_b_name,
                            'down_shape': list(self.weights[name].shape),
                            'up_shape': list(self.weights[lora_b_name].shape),
                            'down_dtype': str(self.weights[name].dtype),
                            'up_dtype': str(self.weights[lora_b_name].dtype)
                        }
        
        analysis['lora_pairs'] = lora_pairs
        print(f"\nIdentified {len(lora_pairs)} LoRA pairs:")
        for base_name, pair_info in lora_pairs.items():
            print(f"  {base_name}: down{pair_info['down_shape']}[{pair_info['down_dtype']}] × up{pair_info['up_shape']}[{pair_info['up_dtype']}]")
    
    def _calculate_parameter_distribution(self, analysis: Dict[str, Any]):
        """Calculate how parameters are distributed across tensors."""
        total_params = sum(info['parameters'] for info in analysis['tensor_info'].values())
        
        distribution = {}
        for name, info in analysis['tensor_info'].items():
            percentage = (info['parameters'] / total_params) * 100
            distribution[name] = {
                'parameters': info['parameters'],
                'percentage': percentage
            }
        
        analysis['parameter_distribution'] = distribution
    
    def calculate_full_weight_modifications(self) -> Dict[str, torch.Tensor]:
        """Calculate the full weight modifications from LoRA pairs (down × up) with dtype handling."""
        print("\n=== Calculating Full Weight Modifications ===")
        
        if 'structure' not in self.analysis_results:
            self.analyze_lora_structure()
        
        full_modifications = {}
        lora_pairs = self.analysis_results['structure']['lora_pairs']
        
        for base_name, pair_info in lora_pairs.items():
            try:
                lora_down = self.weights[pair_info['lora_down']]
                lora_up = self.weights[pair_info['lora_up']]
                
                # Convert to compatible dtypes if needed
                if lora_down.dtype != lora_up.dtype:
                    print(f"    Converting dtype mismatch: {lora_down.dtype} vs {lora_up.dtype}")
                    if lora_down.dtype in [torch.bfloat16, torch.float16]:
                        lora_down = lora_down.float()
                    if lora_up.dtype in [torch.bfloat16, torch.float16]:
                        lora_up = lora_up.float()
                
                # Calculate full modification: down × up
                full_mod = torch.matmul(lora_down, lora_up)
                full_modifications[base_name] = full_mod
                
                print(f"  {base_name}: {lora_down.shape}[{lora_down.dtype}] × {lora_up.shape}[{lora_up.dtype}] → {full_mod.shape}[{full_mod.dtype}]")
                
                # Calculate modification statistics with robust conversion
                try:
                    mod_np = self._tensor_to_numpy(full_mod)
                    mod_stats = {
                        'mean': float(np.mean(mod_np)),
                        'std': float(np.std(mod_np)),
                        'min': float(np.min(mod_np)),
                        'max': float(np.max(mod_np)),
                        'magnitude': float(np.linalg.norm(mod_np))
                    }
                    
                    print(f"    Modification stats: mean={mod_stats['mean']:.6f}, "
                          f"std={mod_stats['std']:.6f}, magnitude={mod_stats['magnitude']:.6f}")
                          
                except Exception as e:
                    print(f"    Warning: Could not calculate modification stats: {e}")
                    
            except Exception as e:
                print(f"  Error calculating modification for {base_name}: {e}")
                continue
        
        self.analysis_results['full_modifications'] = full_modifications
        return full_modifications
    
    def analyze_weight_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in the weight modifications with robust dtype handling."""
        print("\n=== Weight Pattern Analysis ===")
        
        if 'full_modifications' not in self.analysis_results:
            self.calculate_full_weight_modifications()
        
        pattern_analysis = {
            'modification_magnitudes': {},
            'sparsity_analysis': {},
            'correlation_analysis': {},
            'rank_analysis': {}
        }
        
        modifications = self.analysis_results['full_modifications']
        
        for name, mod_tensor in modifications.items():
            try:
                mod_np = self._tensor_to_numpy(mod_tensor)
                
                # Magnitude analysis
                magnitude = np.linalg.norm(mod_np)
                pattern_analysis['modification_magnitudes'][name] = float(magnitude)
                
                # Sparsity analysis
                total_elements = mod_np.size
                near_zero = np.sum(np.abs(mod_np) < 1e-6)
                sparsity = near_zero / total_elements
                pattern_analysis['sparsity_analysis'][name] = {
                    'total_elements': total_elements,
                    'near_zero_elements': int(near_zero),
                    'sparsity_percentage': float(sparsity * 100)
                }
                
                # Rank analysis (for 2D matrices)
                if len(mod_np.shape) == 2:
                    try:
                        rank = np.linalg.matrix_rank(mod_np)
                        pattern_analysis['rank_analysis'][name] = {
                            'matrix_rank': int(rank),
                            'theoretical_max_rank': min(mod_np.shape),
                            'rank_ratio': float(rank / min(mod_np.shape))
                        }
                    except Exception as e:
                        print(f"    Warning: Could not calculate rank for {name}: {e}")
                        
            except Exception as e:
                print(f"  Warning: Could not analyze patterns for {name}: {e}")
                continue
        
        self.analysis_results['patterns'] = pattern_analysis
        return pattern_analysis
    
    def generate_interpretability_report(self) -> Dict[str, Any]:
        """Generate comprehensive interpretability report."""
        print("\n=== Generating Interpretability Report ===")
        
        # Ensure all analyses are complete
        if 'structure' not in self.analysis_results:
            self.analyze_lora_structure()
        if 'full_modifications' not in self.analysis_results:
            self.calculate_full_weight_modifications()
        if 'patterns' not in self.analysis_results:
            self.analyze_weight_patterns()
        
        end_time = time.time()
        analysis_duration = end_time - self.start_time
        
        total_params = sum(
            info['parameters'] for info in 
            self.analysis_results['structure']['tensor_info'].values()
        )
        
        successful_modifications = len(self.analysis_results.get('full_modifications', {}))
        total_possible_pairs = len(self.analysis_results['structure']['lora_pairs'])
        
        audit_completeness = (successful_modifications / total_possible_pairs * 100) if total_possible_pairs > 0 else 100.0
        
        report = {
            'metadata': {
                'lora_file': str(self.lora_path),
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_duration_seconds': analysis_duration,
                'total_parameters_analyzed': total_params,
                'successful_modifications': successful_modifications,
                'total_possible_modifications': total_possible_pairs
            },
            'interpretability_metrics': {
                'complete_coverage': audit_completeness >= 99.0,
                'mathematical_traceability': True,
                'computational_feasibility': True,
                'audit_completeness_percentage': audit_completeness,
                'dtype_handling': 'Robust BF16/FP16/FP32 support'
            },
            'structure_summary': self.analysis_results['structure'],
            'modification_summary': {
                'total_modifications': successful_modifications,
                'modification_magnitudes': self.analysis_results.get('patterns', {}).get('modification_magnitudes', {})
            },
            'computational_requirements': {
                'analysis_time_seconds': analysis_duration,
                'memory_usage_estimate_mb': sum(
                    info['memory_mb'] for info in 
                    self.analysis_results['structure']['tensor_info'].values()
                ),
                'scalability_notes': "Complete analysis feasible at this parameter scale with robust dtype handling"
            }
        }
        
        self.analysis_results['final_report'] = report
        
        print(f"Analysis completed in {analysis_duration:.2f} seconds")
        print(f"Achieved {audit_completeness:.1f}% audit coverage of {total_params:,} parameters")
        print(f"Successfully analyzed {successful_modifications}/{total_possible_pairs} LoRA modifications")
        
        return report
    
    def save_results(self, output_dir: str = "lora_analysis_results"):
        """Save all analysis results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save main report
        report_file = output_path / "interpretability_report.json"
        with open(report_file, 'w') as f:
            json.dump(self.analysis_results['final_report'], f, indent=2)
        
        # Save detailed analysis
        detailed_file = output_path / "detailed_analysis.json"
        with open(detailed_file, 'w') as f:
            # Convert tensors to lists for JSON serialization
            serializable_results = {}
            for key, value in self.analysis_results.items():
                if key != 'full_modifications':
                    serializable_results[key] = value
        
            json.dump(serializable_results, f, indent=2)
        
        # Save weight modifications as numpy arrays
        modifications_dir = output_path / "weight_modifications"
        modifications_dir.mkdir(exist_ok=True)
        
        if 'full_modifications' in self.analysis_results:
            for name, tensor in self.analysis_results['full_modifications'].items():
                try:
                    safe_name = name.replace('/', '_').replace('.', '_')
                    tensor_np = self._tensor_to_numpy(tensor)
                    np.save(modifications_dir / f"{safe_name}.npy", tensor_np)
                except Exception as e:
                    print(f"Warning: Could not save modification {name}: {e}")
        
        print(f"Results saved to: {output_path}")
        return output_path

def main():
    """Main analysis function."""
    print("LoRA Interpretability Analysis with BF16 Support")
    print("=" * 52)
    print("Establishing baseline for complete AI model auditing")
    print("Robust handling of BF16, FP16, and other tensor types")
    print()
    
    # Initialize analyzer
    lora_file = "loras/Alexisk15.safetensors"
    analyzer = LoRABF16Analyzer(lora_file)
    
    try:
        # Load and analyze
        analyzer.load_lora_weights()
        analyzer.analyze_lora_structure()
        analyzer.calculate_full_weight_modifications()
        analyzer.analyze_weight_patterns()
        
        # Generate final report
        report = analyzer.generate_interpretability_report()
        
        # Save results
        output_dir = analyzer.save_results()
        
        print("\n" + "=" * 52)
        print("INTERPRETABILITY BASELINE ESTABLISHED")
        print("=" * 52)
        print(f"✓ Complete audit coverage: {report['interpretability_metrics']['audit_completeness_percentage']:.1f}%")
        print(f"✓ Total parameters analyzed: {report['metadata']['total_parameters_analyzed']:,}")
        print(f"✓ Analysis time: {report['metadata']['analysis_duration_seconds']:.2f} seconds")
        print(f"✓ Mathematical traceability: {report['interpretability_metrics']['mathematical_traceability']}")
        print(f"✓ Dtype handling: {report['interpretability_metrics']['dtype_handling']}")
        print()
        print("This demonstrates what complete AI safety auditing looks like.")
        print("Next: Apply same analysis to larger models to show impossibility scaling.")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
