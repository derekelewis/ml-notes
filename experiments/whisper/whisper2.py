import mlx.core as mx
from mlx.utils import tree_unflatten
from dataclasses import dataclass
from mlx_whisper.whisper import Whisper
from transformers import AutoTokenizer


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


n_mels = 80
n_audio_ctx = 1500
n_audio_state = 384
n_audio_head = 6
n_audio_layer = 4
n_text_ctx = 448
n_text_state = 384
n_text_head = 6
n_text_layer = 4
n_vocab = 51865

modelDimensions = ModelDimensions(
    n_mels=n_mels,
    n_audio_ctx=n_audio_ctx,
    n_audio_state=n_audio_state,
    n_audio_head=n_audio_head,
    n_audio_layer=n_audio_layer,
    n_text_ctx=n_text_ctx,
    n_text_state=n_text_state,
    n_text_head=n_text_head,
    n_text_layer=n_text_layer,
    n_vocab=n_vocab,
)

model = Whisper(dims=modelDimensions)
model_path = "/Users/dlewis/Documents/huggingface/models/mlx-community/whisper-tiny"
weights = mx.load(str(model_path + "/" + "weights.npz"))
weights = tree_unflatten(list(weights.items()))
model.update(weights)
mx.eval(model.parameters())

mx.random.seed(123)
audio_input_tensor = mx.random.normal([1, 2 * n_audio_ctx, n_mels])
print(audio_input_tensor)

mx.random.seed(123)
text_input_tensor = mx.random.randint(0, n_vocab, [1, 1])
print(text_input_tensor)

encoder_output = model.encoder(audio_input_tensor)
decoder_output = model.decoder(text_input_tensor, encoder_output)

print("encoder output:", encoder_output)
print("decoder output:", decoder_output)

tokenizer = AutoTokenizer.from_pretrained("openai/whisper-tiny")
print(tokenizer.decode(decoder_output[0][0][0]))
