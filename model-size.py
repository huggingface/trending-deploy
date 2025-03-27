import requests
import re
import os

# Quantization options mapping
quantization_options = {
    "1-bit": 1,
    "2-bit": 2,
    "3-bit": 3,
    "4-bit": 4,
    "5-bit": 5,
    "6-bit": 6,
    "8-bit": 8,
    "fp16": 16,
    "fp32": 32,
}

class ModelInfo:
    def __init__(self, parameters, quantization=None):
        self.parameters = parameters
        self.quantization = quantization

def get_model_details(repo_id):
    """
    Fetches model details from Hugging Face API.
    
    Args:
        repo_id: The Hugging Face model repository ID
    
    Returns:
        ModelInfo object containing model parameters and quantization info
    """
    try:
        # Get model info using the HF API
        url = f"https://huggingface.co/api/models/{repo_id}"
        
        # Set up headers with authorization if HF_TOKEN is available
        headers = {}
        HF_TOKEN = os.environ.get("HF_TOKEN")
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        model_info = response.json()
        
        # Try to get parameters from safetensors data
        parameters = None
        if model_info.get("safetensors") and model_info["safetensors"].get("total"):
            parameters = int(model_info["safetensors"]["total"])
        # If not in safetensors, try model card metadata
        elif model_info.get("model_card") and model_info["model_card"].get("parameters"):
            parameters = int(model_info["model_card"]["parameters"])
        else:
            # If parameters not in metadata, estimate from model name
            match = re.search(r"(\d+)b", repo_id.lower())
            if match:
                parameters = int(match.group(1)) * 1e9
            else:
                match_m = re.search(r"(\d+)m", repo_id.lower())
                if match_m:
                    parameters = int(match_m.group(1)) * 1e6
                else:
                    raise ValueError('Could not determine model parameters')
        
        # Default to fp32 as most models are distributed in this format
        # Try to determine quantization from safetensors data
        quantization = 'fp32'
        if model_info.get("safetensors") and model_info["safetensors"].get("parameters"):
            # Check for different precision formats in parameters
            if "BF16" in model_info["safetensors"]["parameters"]:
                quantization = "fp16"  # BF16 is similar to FP16 in size
            elif "FP16" in model_info["safetensors"]["parameters"]:
                quantization = "fp16"
            # Add more quantization detection if needed
        
        return ModelInfo(parameters, quantization)
    
    except requests.exceptions.RequestException as error:
        print(f'Error in get_model_details: {error}')
        raise
    except Exception as error:
        print(f'Error processing model details: {error}')
        raise

def calculate_memory_usage(parameters_in_billions, quantization, context_window, os_overhead_gb=2):
    """
    Calculates the expected memory usage of an LLM in GB.
    
    Args:
        parameters_in_billions: Number of parameters in the model (in billions)
        quantization: Quantization level (e.g., "4-bit", "fp16")
        context_window: Size of the context window in tokens
        os_overhead_gb: OS overhead in GB (default is 2 GB)
    
    Returns:
        The total memory usage in GB
    """
    # Convert parameters from billions to actual count
    parameters = parameters_in_billions * 1e9
    
    # Calculate memory for parameters
    bits_per_parameter = quantization_options[quantization]
    bytes_per_parameter = bits_per_parameter / 8
    parameter_memory_bytes = parameters * bytes_per_parameter
    
    # Calculate memory for context window
    context_memory_bytes = context_window * 0.5 * 1e6  # 0.5 bytes per token
    
    # Total memory in bytes
    total_memory_bytes = parameter_memory_bytes + context_memory_bytes + os_overhead_gb * 1e9
    
    # Convert to GB
    total_memory_gb = total_memory_bytes / 1e9
    
    return total_memory_gb

def calculate_huggingface_model_memory(repo_id, context_window, os_overhead_gb=2):
    """
    Calculates memory usage for a Hugging Face model.
    
    Args:
        repo_id: The Hugging Face model repository ID
        context_window: Size of the context window in tokens
        os_overhead_gb: OS overhead in GB (default is 2 GB)
    
    Returns:
        The calculated memory usage in GB
    """
    model_info = get_model_details(repo_id)

    print(model_info.parameters)
    
    # Convert parameters to billions
    parameters_in_billions = model_info.parameters / 1e9
    
    # Use default fp32 if no quantization specified
    quantization = model_info.quantization or 'fp32'
    
    return calculate_memory_usage(parameters_in_billions, quantization, context_window, os_overhead_gb)

def run_tests():
    """Test function to demonstrate usage"""
    print('Running memory calculation tests...\n')
    
    # Test 1: Direct memory calculation
    direct_test = calculate_memory_usage(7, "4-bit", 2048)
    print('Test 1: Direct Memory Calculation')
    print('Model: 7B parameters, 4-bit quantization, 2048 context window')
    print(f'Expected Memory Usage: {direct_test:.2f} GB\n')
    
    # Test 2: Hugging Face model calculation
    try:
        print('Test 2: Hugging Face Model Calculation')
        print('Model: Qwen/Qwen2.5-1.5B-Instruct, 2048 context window')
        hf_test = calculate_huggingface_model_memory('Qwen/Qwen2.5-1.5B-Instruct', 2048)
        print(f'Expected Memory Usage: {hf_test:.2f} GB\n')
    except Exception as error:
        print(f'Error fetching Hugging Face model details: {error}')

if __name__ == "__main__":
    run_tests()
