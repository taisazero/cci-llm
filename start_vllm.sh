#!/bin/bash

# start_server.bash - Interactive LLM Server Launcher with vLLM
# Usage: ./start_server.bash [model_name] [api_key]

set -e

source /home/ealhossa/miniconda3/etc/profile.d/conda.sh
# Activate vLLM environment
source activate vllm
echo "Environment loaded successfully"

# Detect number of GPUs
num_gpus=$(nvidia-smi -L | wc -l)
echo "Detected $num_gpus GPU(s)"

# Configuration
WEIGHTS_DIR="/home/weights"
TMUX_SESSION_NAME="vllm"
DEFAULT_PORT=8000

# Global variables
API_KEY=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if tmux session exists
tmux_session_exists() {
    tmux has-session -t "$1" 2>/dev/null
}

# Function to kill existing tmux session
kill_tmux_session() {
    if tmux_session_exists "$1"; then
        print_info "Killing existing tmux session: $1"
        tmux kill-session -t "$1"
    fi
}

# Function to start tmux session with improved robustness
start_tmux_session() {
    local session_name="$1"
    local command="$2"
    
    print_info "Starting tmux session: $session_name"
    
    # Kill any existing tmux session first
    kill_tmux_session "$session_name"
    
    # Create tmux session with proper command execution
    # Use bash -c to ensure the command runs properly and keeps the session alive
    tmux new-session -d -s "$session_name" bash -c "source activate vllm; $command; exec bash"
    
    # Wait a moment for the session to initialize
    sleep 5
    
    # Check if session is still running
    if tmux_session_exists "$session_name"; then
        print_success "Started tmux session: $session_name"
        print_info "To attach to the session, run: tmux attach -t $session_name"
        print_info "To detach from the session, press Ctrl+B then D"
    else
        print_error "Failed to start tmux session"
        return 1
    fi
}

# Function to get available local models
get_local_models() {
    local models=()
    if [ -d "$WEIGHTS_DIR" ]; then
        for dir in "$WEIGHTS_DIR"/*; do
            local dirname=$(basename "$dir")
            if [ -d "$dir" ] && [ "$dirname" != "server" ] && [ "$dirname" != "lost+found" ]; then
                models+=("$dirname")
            fi
        done
    fi
    echo "${models[@]}"
}

# Function to check if model exists locally
model_exists_locally() {
    local model_name="$1"
    [ -d "$WEIGHTS_DIR/$model_name" ]
}

# Function to construct vLLM serve command with API key support
construct_vllm_command() {
    local model="$1"
    local is_local="$2"
    local api_key="$3"
    
    local cmd="vllm serve '$model'"
    cmd="$cmd --dtype auto"
    cmd="$cmd --port $DEFAULT_PORT"
    cmd="$cmd --host 0.0.0.0"
    cmd="$cmd --trust-remote-code"
    cmd="$cmd --disable-log-requests"
    
    # Add served model name using the name after the weight directory
    local served_model_name
    if [ "$is_local" = true ]; then
        # For local models, use the basename of the model path
        served_model_name=$(basename "$model")
    else
        # For remote models, use the model name as-is
        served_model_name="$model"
    fi
    cmd="$cmd --served-model-name '$served_model_name'"
    
    # Add API key if provided
    if [ -n "$api_key" ]; then
        cmd="$cmd --api-key '$api_key'"
        print_info "API key authentication enabled"
    fi
    
    # Add GPU memory utilization for 70B models
    if [[ "$model" =~ [70][0-9]*B ]] || [[ "$model" =~ [70][0-9]*b ]]; then
        cmd="$cmd --gpu-memory-utilization 0.95"
        print_info "Detected 70B+ model - using GPU memory utilization 0.95"
    fi
    
    # Add GPTQ optimizations for models with GPTQ in the name
    if [[ "$model" =~ [Gg][Pp][Tt][Qq] ]]; then
        cmd="$cmd --quantization gptq"
        cmd="$cmd --dtype float16"
        print_info "Detected GPTQ model - using GPTQ quantization and float16 dtype"
    fi
    
    # Add specific optimizations for Llama-3.3-70B
    if [[ "$model" =~ [Ll]lama-3\.3-70[Bb] ]]; then
        # Only add these if not already added by GPTQ detection
        if [[ ! "$model" =~ [Gg][Pp][Tt][Qq] ]]; then
            cmd="$cmd --quantization gptq"
            cmd="$cmd --dtype float16"
        fi
        cmd="$cmd --max-model-len 16384"
        cmd="$cmd --enforce-eager"
        print_info "Detected Llama-3.3-70B - using GPTQ quantization, max model length 16384, float16 dtype, and enforce eager execution"
    fi
    
    # Use tensor parallelism if multiple GPUs are available
    if [ "$num_gpus" -gt 1 ]; then
        cmd="$cmd --tensor-parallel-size $num_gpus"
        print_info "Using tensor parallelism across $num_gpus GPUs"
    fi
    
    # If model is not local, set download directory
    if [ "$is_local" = false ]; then
        cmd="$cmd --download-dir '$WEIGHTS_DIR'"
        print_info "Downloaded models will be saved to $WEIGHTS_DIR with full permissions"
    fi
    
    echo "$cmd"
}

# Function to prompt for API key
prompt_api_key() {
    echo
    print_info "=== API Key Configuration ==="
    print_info "You can optionally set an API key for authentication."
    print_info "Leave empty for no authentication (public access)."
    echo
    read -p "Enter API key (or press Enter to skip): " input_api_key
    
    if [ -n "$input_api_key" ]; then
        API_KEY="$input_api_key"
        print_info "API key set successfully"
    else
        print_info "No API key set - server will be publicly accessible"
    fi
    echo
}

# Function to display interactive menu
show_menu() {
    local models=($(get_local_models))
    
    echo
    print_info "=== vLLM Server Launcher ==="
    echo
    
    if [ ${#models[@]} -eq 0 ]; then
        print_warning "No local models found in $WEIGHTS_DIR"
    else
        echo "Available local models:"
        for i in "${!models[@]}"; do
            printf "  %d) %s\n" $((i+1)) "${models[$i]}"
        done
        echo
    fi
    
    local next_option=$((${#models[@]} + 1))
    echo "Other options:"
    printf "  %d) Enter HuggingFace model name manually\n" $next_option
    printf "  %d) Exit\n" $((next_option + 1))
    echo
    
    while true; do
        read -p "Select an option: " choice
        
        if [[ "$choice" =~ ^[0-9]+$ ]]; then
            if [ "$choice" -ge 1 ] && [ "$choice" -le ${#models[@]} ]; then
                # Local model selected
                selected_model="${models[$((choice-1))]}"
                prompt_api_key
                
                # Check if this is a models-- directory (downloaded HuggingFace model)
                if [[ "$selected_model" =~ ^models-- ]]; then
                    # Convert back to HuggingFace format and serve with download directory
                    hf_model_name=$(convert_models_dir_to_hf_name "$selected_model")
                    print_info "Detected HuggingFace model directory: $selected_model"
                    print_info "Converting to HuggingFace format: $hf_model_name"
                    print_info "Serving with download directory: $WEIGHTS_DIR"
                    serve_model "$hf_model_name" false "$API_KEY"
                else
                    # Regular local model
                    serve_model "$WEIGHTS_DIR/$selected_model" true "$API_KEY"
                fi
                break
            elif [ "$choice" -eq $next_option ]; then
                # Manual entry
                read -p "Enter HuggingFace model name (e.g., meta-llama/Meta-Llama-3-8B-Instruct): " manual_model
                if [ -n "$manual_model" ]; then
                    prompt_api_key
                    serve_model "$manual_model" false "$API_KEY"
                    break
                else
                    print_error "Model name cannot be empty"
                fi
            elif [ "$choice" -eq $((next_option + 1)) ]; then
                # Exit
                print_info "Exiting..."
                exit 0
            else
                print_error "Invalid selection. Please try again."
            fi
        else
            print_error "Please enter a valid number."
        fi
    done
}

# Function to serve the model with API key support
serve_model() {
    local model="$1"
    local is_local="$2"
    local api_key="$3"
    
    print_info "Selected model: $model"
    
    # Check if model exists locally if it's supposed to be local
    if [ "$is_local" = true ] && ! model_exists_locally "$(basename "$model")"; then
        print_error "Local model directory not found: $model"
        exit 1
    fi
    
    # Kill existing tmux session if it exists
    kill_tmux_session "$TMUX_SESSION_NAME"
    
    # Wait a moment before starting new session
    sleep 5
    
    # Construct vLLM command
    local vllm_cmd=$(construct_vllm_command "$model" "$is_local" "$api_key")
    
    print_info "vLLM command: $vllm_cmd"
    
    # Create weights directory if it doesn't exist with proper permissions
    mkdir -p "$WEIGHTS_DIR"
    
    # Start tmux session with vLLM server
    if start_tmux_session "$TMUX_SESSION_NAME" "$vllm_cmd"; then
        
        # If it's a remote model, set permissions after potential download
        if [ "$is_local" = false ]; then
            print_info "Setting permissions for downloaded model files..."
            
            # Convert model name to HuggingFace directory format (org/model -> models--org--model)
            local hf_dir_name="models--$(echo "$model" | sed 's/\//--/g')"
            local model_download_dir="$WEIGHTS_DIR/$hf_dir_name"
            # If the model directory is not found, print a warning
            if [ ! -d "$model_download_dir" ]; then
                print_warning "Please wait for the model to be downloaded..."
                print_warning "This may take a few minutes..."
                print_warning "Once the model is downloaded, please set permissions by running: chmod 777 -R $model_download_dir"
                # Make sure the user sees the warning
                sleep 3
            fi
        fi
        echo
        print_success "vLLM server started successfully!"
        print_info "Server will be available at: http://localhost:$DEFAULT_PORT"
        print_info "API endpoint: http://localhost:$DEFAULT_PORT/v1"
        if [ -n "$api_key" ]; then
            print_info "API key authentication is enabled"
            print_info "Include 'Authorization: Bearer $api_key' in your requests"
        fi
        if [ "$num_gpus" -gt 1 ]; then
            print_info "Using tensor parallelism across $num_gpus GPUs"
        else
            print_info "Using single GPU"
        fi
        echo
        print_info "To monitor the server logs and check when the server is ready:"
        print_info "  tmux attach -t $TMUX_SESSION_NAME"
        print_info "The server will take a few minutes to start..."
        echo
        print_info "To stop the server:"
        print_info "  tmux kill-session -t $TMUX_SESSION_NAME"
        echo
    else
        print_error "Failed to start vLLM server"
        exit 1
    fi
}

# Function to show help information
show_help() {
    echo
    print_info "=== vLLM Server Launcher Help ==="
    echo
    echo "DESCRIPTION:"
    echo "  Interactive LLM server launcher using vLLM for high-performance inference."
    echo "  Supports both local models and automatic downloading from HuggingFace."
    echo
    echo "USAGE:"
    echo "  $0 [OPTIONS] [model_name] [api_key]"
    echo
    echo "OPTIONS:"
    echo "  -h, --help              Show this help message and exit"
    echo
    echo "ARGUMENTS:"
    echo "  model_name              Name of the model to serve:"
    echo "                          - Local model: directory name in $WEIGHTS_DIR"
    echo "                          - HuggingFace: format 'org/model-name'"
    echo "  api_key                 Optional API key for authentication"
    echo
    echo "EXAMPLES:"
    echo "  $0                                          # Interactive menu"
    echo "  $0 -h                                       # Show help"
    echo "  $0 Llama-3.3-70B-Instruct                  # Serve local model"
    echo "  $0 meta-llama/Meta-Llama-3-8B-Instruct     # Download and serve from HF"
    echo "  $0 Llama-3.3-70B-Instruct my-secret-key    # Serve with API key"
    echo
    echo "FEATURES:"
    echo "  • Auto-detection of GPU count for tensor parallelism"
    echo "  • Optimized settings for 70B+ models (high GPU memory utilization)"
    echo "  • GPTQ quantization support for compatible models"
    echo "  • Special optimizations for Llama-3.3-70B models"
    echo "  • Automatic model downloading with proper permissions"
    echo "  • tmux session management for background execution"
    echo "  • Optional API key authentication"
    echo
    echo "SERVER INFO:"
    echo "  • Default port: $DEFAULT_PORT"
    echo "  • API endpoint: http://localhost:$DEFAULT_PORT/v1"
    echo "  • Models directory: $WEIGHTS_DIR"
    echo "  • tmux session: $TMUX_SESSION_NAME"
    echo
    echo "MANAGEMENT COMMANDS:"
    echo "  tmux attach -t $TMUX_SESSION_NAME           # Monitor server logs"
    echo "  tmux kill-session -t $TMUX_SESSION_NAME     # Stop server"
    echo "  tmux list-sessions                          # List all tmux sessions"
    echo
    exit 0
}

# Function to validate model name format
validate_model_name() {
    local model="$1"
    
    # Check if it's a HuggingFace format (org/model) or local model name
    if [[ "$model" =~ ^[a-zA-Z0-9_-]+/[a-zA-Z0-9._-]+$ ]] || [[ "$model" =~ ^[a-zA-Z0-9._-]+$ ]]; then
        return 0
    else
        return 1
    fi
}

# Function to convert models-- directory name back to HuggingFace format
convert_models_dir_to_hf_name() {
    local dir_name="$1"
    
    # Check if directory starts with models--
    if [[ "$dir_name" =~ ^models-- ]]; then
        # Remove "models--" prefix and convert "--" back to "/"
        echo "${dir_name#models--}" | sed 's/--/\//g'
        return 0
    else
        # Not a models-- directory, return as-is
        echo "$dir_name"
        return 1
    fi
}

# Main function
main() {
    # Check for help option first
    for arg in "$@"; do
        if [ "$arg" = "-h" ] || [ "$arg" = "--help" ]; then
            show_help
        fi
    done
    
    # Check if tmux is installed
    if ! command -v tmux &> /dev/null; then
        print_error "tmux is not installed. Please install tmux first."
        exit 1
    fi
    
    # Check if vllm is installed
    if ! command -v vllm &> /dev/null; then
        print_error "vLLM is not installed. Please install vLLM first."
        print_info "Install with: pip install vllm"
        exit 1
    fi
    
    # Parse command line arguments
    if [ $# -eq 1 ]; then
        # Only model name provided
        local model_arg="$1"
        
        if ! validate_model_name "$model_arg"; then
            print_error "Invalid model name format: $model_arg"
            exit 1
        fi
        
        # Check if it's a local model
        if model_exists_locally "$model_arg"; then
            print_info "Found local model: $model_arg"
            serve_model "$WEIGHTS_DIR/$model_arg" true ""
        else
            print_info "Model not found locally, will download from HuggingFace: $model_arg"
            serve_model "$model_arg" false ""
        fi
    elif [ $# -eq 2 ]; then
        # Both model name and API key provided
        local model_arg="$1"
        local api_key_arg="$2"
        
        if ! validate_model_name "$model_arg"; then
            print_error "Invalid model name format: $model_arg"
            exit 1
        fi
        
        API_KEY="$api_key_arg"
        
        # Check if it's a local model
        if model_exists_locally "$model_arg"; then
            print_info "Found local model: $model_arg"
            serve_model "$WEIGHTS_DIR/$model_arg" true "$API_KEY"
        else
            print_info "Model not found locally, will download from HuggingFace: $model_arg"
            serve_model "$model_arg" false "$API_KEY"
        fi
    elif [ $# -eq 0 ]; then
        # No arguments, show interactive menu
        show_menu
    else
        print_error "Usage: $0 [OPTIONS] [model_name] [api_key]"
        print_info "Use '$0 --help' for detailed information and examples."
        exit 1
    fi
}

# Run main function with all arguments
main "$@" 