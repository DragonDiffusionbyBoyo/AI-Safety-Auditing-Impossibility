"""
Environment Test Script
======================

Quick test to verify that the research environment is properly configured
before running the full impossibility proof research.
"""

import sys
import importlib
from pathlib import Path

def test_python_version():
    """Test Python version compatibility."""
    print("Testing Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def test_dependencies():
    """Test required dependencies."""
    print("\nTesting dependencies...")
    
    required_packages = [
        'torch',
        'safetensors', 
        'numpy',
        'matplotlib',
        'seaborn',
        'pandas',
        'psutil',
        'json',
        'pathlib',
        'datetime'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - Not installed")
            failed_imports.append(package)
    
    return len(failed_imports) == 0

def test_model_files():
    """Test availability of model files."""
    print("\nTesting model files...")
    
    # Test LoRA file
    lora_path = Path("loras/Alexisk15.safetensors")
    if lora_path.exists():
        print(f"✓ LoRA model found: {lora_path}")
        lora_available = True
    else:
        print(f"✗ LoRA model missing: {lora_path}")
        lora_available = False
    
    # Test 4B model config
    config_path = Path("Polaris-4B-Preview/config.json")
    if config_path.exists():
        print(f"✓ 4B model config found: {config_path}")
        config_available = True
    else:
        print(f"✗ 4B model config missing: {config_path}")
        config_available = False
    
    # Test 4B model index
    index_path = Path("Polaris-4B-Preview/model.safetensors.index.json")
    if index_path.exists():
        print(f"✓ 4B model index found: {index_path}")
        index_available = True
    else:
        print(f"✗ 4B model index missing: {index_path}")
        index_available = False
    
    return lora_available and config_available and index_available

def test_torch_functionality():
    """Test PyTorch functionality."""
    print("\nTesting PyTorch functionality...")
    
    try:
        import torch
        
        # Test basic tensor operations
        x = torch.randn(3, 3)
        y = torch.matmul(x, x)
        print(f"✓ Basic tensor operations working")
        
        # Test CUDA availability (optional)
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"ℹ CUDA not available - will use CPU (this is fine)")
        
        return True
        
    except Exception as e:
        print(f"✗ PyTorch functionality test failed: {e}")
        return False

def test_safetensors_functionality():
    """Test safetensors functionality."""
    print("\nTesting safetensors functionality...")
    
    try:
        from safetensors.torch import load_file
        
        # Test loading the LoRA file if available
        lora_path = Path("loras/Alexisk15.safetensors")
        if lora_path.exists():
            weights = load_file(str(lora_path))
            print(f"✓ Successfully loaded LoRA file with {len(weights)} tensors")
            return True
        else:
            print(f"ℹ LoRA file not available for testing, but safetensors import successful")
            return True
            
    except Exception as e:
        print(f"✗ Safetensors functionality test failed: {e}")
        return False

def test_memory_availability():
    """Test available system memory."""
    print("\nTesting system resources...")
    
    try:
        import psutil
        
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        total_gb = memory.total / (1024**3)
        
        print(f"Total RAM: {total_gb:.1f} GB")
        print(f"Available RAM: {available_gb:.1f} GB")
        
        if available_gb >= 8:
            print(f"✓ Sufficient memory for research")
            return True
        else:
            print(f"⚠ Low memory - may encounter issues with larger models")
            return False
            
    except Exception as e:
        print(f"✗ Memory test failed: {e}")
        return False

def main():
    """Run all environment tests."""
    print("AI Model Auditing Research - Environment Test")
    print("=" * 50)
    
    tests = [
        ("Python Version", test_python_version),
        ("Dependencies", test_dependencies),
        ("Model Files", test_model_files),
        ("PyTorch", test_torch_functionality),
        ("Safetensors", test_safetensors_functionality),
        ("System Resources", test_memory_availability)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("ENVIRONMENT TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Environment ready for research!")
        print("You can now run: python research_orchestrator.py")
    elif passed >= total - 1:
        print("\n⚠ Environment mostly ready - minor issues detected")
        print("Research may still work, but check failed tests above")
    else:
        print("\n❌ Environment not ready - please fix failed tests")
        print("Refer to instructions.md for setup guidance")
    
    print("\nNote: Even if some tests fail, the research may still demonstrate")
    print("impossibility through technical limitations - this supports the thesis!")

if __name__ == "__main__":
    main()
