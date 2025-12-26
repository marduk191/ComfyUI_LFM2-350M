from .nodes import LFM2Loader, LFM2Generator

NODE_CLASS_MAPPINGS = {
    "LFM2Loader": LFM2Loader,
    "LFM2Generator": LFM2Generator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LFM2Loader": "LiquidAI LFM-2-350M Loader",
    "LFM2Generator": "LiquidAI LFM-2-350M Generator"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
