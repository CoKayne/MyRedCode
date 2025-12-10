import os
from config import get_config
from evaluation import evaluate_model

def select_best_gpu():
    """
    Automatically select the GPU with the most free memory.
    Sets CUDA_VISIBLE_DEVICES if GPUtil is available and GPUs are detected.
    """
    # Skip if CUDA_VISIBLE_DEVICES is already set
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    
    try:
        from GPUtil import getGPUs
        gpus = getGPUs()
        if gpus:
            best = max(gpus, key=lambda g: g.memoryFree)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best.id)
            print(f"Selected GPU {best.id} (Free Memory: {best.memoryFree:.0f} MB)")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            print("No GPUs detected, using CPU instead")
    except ImportError:
        # GPUtil not installed, skip auto-selection
        pass
    except Exception as e:
        print(f"Warning: Could not auto-select GPU: {e}")

def main():
    # Auto-select best GPU before loading models
    select_best_gpu()
    
    config = get_config()
    evaluate_model(config)

if __name__ == "__main__":
    main()
