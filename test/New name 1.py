import os
import sys
import torch
from scipy.io import wavfile
import folder_paths # ComfyUI's path manager
import numpy as np

# --- Path Setup ---
# Add the node's root directory to sys.path to allow imports of the copied RVC modules
node_root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, node_root_dir)

# Now we can import from the copied modules
from configs.config import Config
from infer.modules.vc.modules import VC

# --- ComfyUI-specific Path Definitions ---
# Get the main ComfyUI folder paths
comfy_models_dir = os.path.join(folder_paths.models_dir)
comfy_base_dir = os.path.abspath(os.path.join(comfy_models_dir, ".."))

# Define our custom directories as requested by the user
RVC_MODELS_DIR = os.path.join(comfy_models_dir, "RVC")
RVC_INDEX_DIR = os.path.join(comfy_base_dir, "RVC", "index")

# --- Model Caching ---
vc_model_cache = {}
config_cache = None

def get_config():
    """Initializes and caches the RVC Config."""
    global config_cache
    if config_cache is None:
        config = Config()
        # You might need to adjust these device settings depending on your system
        # config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # config.is_half = True if torch.cuda.is_available() else False
        config_cache = config
    return config_cache

def get_vc_model(model_name):
    """Loads and caches the RVC voice model."""
    if model_name in vc_model_cache:
        return vc_model_cache[model_name]
    
    print(f"RVC: Loading model '{model_name}'...")
    config = get_config()
    vc = VC(config)
    
    # get_vc expects the model name to load it from assets/weights, 
    # but we need to load from an absolute path. We'll temporarily override.
    original_get_vc = vc.get_vc
    
    try:
        # A bit of a hack: The get_vc method is hardcoded to look in 'assets/weights'.
        # We can't easily change it without modifying the library code.
        # Instead, we pass the full path as the "name" and handle it inside.
        # This part of the original code is not flexible, so we work around it.
        model_path = os.path.join(RVC_MODELS_DIR, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        vc.get_vc(model_path, True) # Pass the full path and a flag to indicate it's a direct path
    finally:
        vc.get_vc = original_get_vc # Restore original method
        
    vc_model_cache[model_name] = vc
    print(f"RVC: Model '{model_name}' loaded successfully.")
    return vc

# --- Helper function to find files and create dirs ---
def find_files(directory, extensions):
    """Finds files with given extensions in a directory. Creates dir if it doesn't exist."""
    if not os.path.exists(directory):
        print(f"RVC: Creating directory: {directory}")
        os.makedirs(directory)
        return []
    
    files = [f for f in os.listdir(directory) if any(f.endswith(ext) for ext in extensions)]
    if not files:
        print(f"RVC: No files with extensions {extensions} found in {directory}. Please add your files.")
    return files

# --- The Main Node Class ---
class RVC_Voice_Conversion:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()

    @classmethod
    def INPUT_TYPES(s):
        # Scan the user-defined directories for models and index files
        model_names = find_files(RVC_MODELS_DIR, [".pth"])
        index_files = [""] + find_files(RVC_INDEX_DIR, [".index"])

        return {
            "required": {
                "model_name": (model_names,),
                "input_audio_path": ("STRING", {"default": "path/to/your/audio.wav"}),
                "f0up_key": ("INT", {"default": 0, "min": -24, "max": 24, "step": 1, "display": "slider"}),
                "f0method": (["rmvpe", "pm", "harvest"],), # rmvpe is generally best
                "index_path": (index_files,),
                "index_rate": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "filter_radius": ("INT", {"default": 3, "min": 0, "max": 10, "step": 1, "display": "slider"}),
                "resample_sr": ("INT", {"default": 0, "min": 0, "max": 48000, "step": 1}),
                "rms_mix_rate": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "protect": ("FLOAT", {"default": 0.33, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_audio_path", "status_message")
    FUNCTION = "convert_voice"
    CATEGORY = "RVC"

    def convert_voice(self, model_name, input_audio_path, f0up_key, f0method, index_path, index_rate, filter_radius, resample_sr, rms_mix_rate, protect):
        try:
            # --- Validate Inputs ---
            if not os.path.exists(input_audio_path):
                return (None, f"Error: Input audio file not found at {input_audio_path}")

            model_path = os.path.join(RVC_MODELS_DIR, model_name)
            if not os.path.exists(model_path):
                 return (None, f"Error: Model file not found. Ensure '{model_name}' is in '{RVC_MODELS_DIR}'")

            absolute_index_path = os.path.join(RVC_INDEX_DIR, index_path) if index_path else ""
            if index_path and not os.path.exists(absolute_index_path):
                return (None, f"Error: Index file not found. Ensure '{index_path}' is in '{RVC_INDEX_DIR}'")

            # --- Load Model and Config ---
            # The original vc.get_vc is a bit tricky. It's designed to load from a specific folder structure.
            # We need to modify the VC class slightly to allow loading from a direct path.
            # Let's modify the class on the fly. This is hacky but avoids changing the library code.
            
            def get_vc_patched(self, name_or_path, is_absolute_path=False):
                if is_absolute_path:
                    print(f"Loading weights from direct path: {name_or_path}")
                    ckpt = torch.load(name_or_path, map_location="cpu")
                else: # Original behavior
                    name_or_path = os.path.join(self.config.weights_dir, name_or_path)
                    print(f"Loading weights from: {name_or_path}")
                    ckpt = torch.load(name_or_path, map_location="cpu")
                self.cpt = ckpt
                self.t_config = self.cpt.get("config", self.config)
                self.t_config.device = self.config.device
                self.t_config.is_half = self.config.is_half
                self.net_g = self.get_net_g()
                self.is_half = self.config.is_half
                self.cpt = None
                # torch.cuda.empty_cache() # Optional cleanup

            # Apply the patch
            VC.get_vc = get_vc_patched
            
            config = get_config()
            vc = VC(config)
            vc.get_vc(model_path, True) # Load model with absolute path

            # --- Perform Inference ---
            print("RVC: Starting voice conversion...")
            status_message, (target_sr, audio_opt) = vc.vc_single(
                sid=0,
                input_audio_path=input_audio_path,
                f0_up_key=f0up_key,
                f0_file=None,
                f0_method=f0method,
                file_index=absolute_index_path,
                file_index2=None,
                index_rate=index_rate,
                filter_radius=filter_radius,
                resample_sr=resample_sr,
                rms_mix_rate=rms_mix_rate,
                protect=protect,
            )
            
            print(f"RVC Status: {status_message}")
            if "Success" not in status_message:
                return(None, f"RVC processing failed: {status_message}")

            # --- Save Output ---
            filename_prefix = os.path.splitext(os.path.basename(input_audio_path))[0]
            output_filename = f"RVC_{filename_prefix}_{os.path.splitext(model_name)[0]}.wav"
            output_path = os.path.join(self.output_dir, output_filename)
            
            wavfile.write(output_path, target_sr, audio_opt.astype(np.int16))
            print(f"RVC: Saved output to {output_path}")

            return (output_path, "Success")

        except Exception as e:
            error_message = f"RVC Node Error: {e}"
            print(f"\n{error_message}")
            import traceback
            traceback.print_exc()
            return (None, error_message)
