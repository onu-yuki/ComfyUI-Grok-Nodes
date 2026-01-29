
from .grok_image_node import GrokImage
from .grok_chat_to_start_end_image_prompts_node import GrokChatToStartEndImagePrompts
from .grok_chat_to_image_prompt_node import GrokChatToImagePrompt
from .grok_imagine_video_node import GrokImagineVideo
from .grok_imagine_image_node import GrokImagineImage

NODE_CLASS_MAPPINGS = {
    "GrokImage": GrokImage,
    "GrokChatToStartEndImagePrompts": GrokChatToStartEndImagePrompts,
    "GrokChatToImagePrompt": GrokChatToImagePrompt,
    "GrokImagineVideo": GrokImagineVideo,
    "GrokImagineImage": GrokImagineImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrokImageNode": "Grok Image",
    "GrokChatToStartEndImagePrompts": "Grok Chat → Image Prompts (Start/End)",
    "GrokChatToImagePrompt": "Grok Chat → Image Prompt",
    "GrokImagineVideo": "Grok Imagine Video",
    "GrokImagineImage": "Grok Imagine Image"
}
