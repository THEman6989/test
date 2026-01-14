# This file makes the ComfyUI_RVC folder a Python package
# and registers the custom node with ComfyUI.

# We lazy-load the node to prevent issues with missing dependencies
# or other startup problems.
def get_node_class(module_name, node_name):
    from importlib import util
    import traceback
    
    module_path = f".{module_name}"
    
    try:
        module_spec = util.find_spec(module_path, package=__name__)
        if module_spec:
            module = util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            if hasattr(module, node_name):
                print(f"Successfully loaded {node_name} from {module_name}")
                return getattr(module, node_name)
            else:
                print(f"[ERROR] Failed to load {node_name} from {module_name}: Class not found.")
                return None
        else:
            print(f"[ERROR] Failed to find module {module_name}.")
            return None
    except Exception as e:
        print(f"[ERROR] Failed to load {node_name} from {module_name}:\n{traceback.format_exc()}")
        return None

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

RVCNode = get_node_class("rvc_node", "RVC_Voice_Conversion")

if RVCNode:
    NODE_CLASS_MAPPINGS["RVC Voice Conversion"] = RVCNode
    NODE_DISPLAY_NAME_MAPPINGS["RVC Voice Conversion"] = "RVC Voice Conversion"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
