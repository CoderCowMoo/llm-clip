import llm
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModel, AutoTokenizer
import torch
import io


@llm.hookimpl
def register_embedding_models(register):
    register(ClipEmbeddingModel())
    register(SigLIPEmbeddingModel())


class ClipEmbeddingModel(llm.EmbeddingModel):
    model_id = "clip"
    supports_binary = True
    supports_text = True

    def __init__(self):
        self._model = None

    def embed_batch(self, items):
        # Embeds a mix of text strings and binary images
        if self._model is None:
            self._model = SentenceTransformer("clip-ViT-B-32")

        to_embed = []

        for item in items:
            if isinstance(item, bytes):
                # If the item is a byte string, treat it as image data and convert to Image object
                to_embed.append(Image.open(io.BytesIO(item)))
            elif isinstance(item, str):
                to_embed.append(item)

        embeddings = self._model.encode(to_embed)
        return [[float(num) for num in embedding] for embedding in embeddings]
    
# Siglip model using huggingface transformers
class SigLIPEmbeddingModel(llm.EmbeddingModel):
    model_id = "siglip"
    supports_binary = True
    supports_text = True

    def __init__(self) -> None:
        self._model = None
    
    def embed_batch(self, items):
        if self._model is None:
            # we use the model for extracting features
            self._model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
            # and processor and tokenizer for processing image and text data.
            self._processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
            self._tokenizer = AutoTokenizer.from_pretrained("google/siglip-so400m-patch14-384")
        
        images_to_embed = []
        strings_to_embed = []
        for item in items:
            # strings are appended
            if isinstance(item, str):
                strings_to_embed.append(item)
            elif isinstance(item, bytes):
                images_to_embed.append(Image.open(io.BytesIO(item)))
        t_inputs = self._tokenizer(strings_to_embed, return_tensors="pt", padding="max_length") if strings_to_embed else None
        i_inputs = self._processor(images=images_to_embed, return_tensors="pt") if images_to_embed else None
        with torch.no_grad():
            outputs = []
            outputs.append(self._model.get_text_features(**t_inputs)) if t_inputs else None
            outputs.append(self._model.get_image_features(**i_inputs)) if i_inputs else None
        # output format is 2D list of floats so we need to flatten it and squeeze it.
        outputs = [torch.squeeze(output).numpy().tolist() for output in outputs if output is not None]
        return outputs