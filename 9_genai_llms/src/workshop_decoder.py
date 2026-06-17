from __future__ import annotations

"""Small decoder-only transformer used by the GenAI workshop notebook.

The notebook should show the LLM training loop without hiding everything behind
one large framework call. This module keeps the reusable pieces in one place:
text preparation, token/id conversion, training-window creation, the Keras
decoder model, and generation helpers.

The implementation is intentionally compact. It is not trying to compete with a
foundation model; it is a readable version of the same core idea: represent text
as token ids, learn to predict the next token, then generate by repeating that
prediction step.
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


TOKEN_PATTERN = r"[a-zA-Z0-9]+|[.,:;!?()/-]"
SPECIAL_TOKENS = ["<pad>", "<unk>"]


def clean_text(text: str) -> str:
    """Convert repository text into a cleaner training corpus.

    The repository contains markdown links, headings, code fences, image syntax,
    and notebook formatting. Those characters are useful for documentation, but
    they add noise when the model is learning simple language patterns. This
    function removes the most distracting markup and normalizes whitespace while
    keeping the words and punctuation that should become tokens.
    """
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[#>*_`|]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_workshop_source(relative_path: str) -> bool:
    """Decide whether a repository file belongs in the language-model corpus.

    The training text should come from participant-facing workshop materials,
    not from Git internals, Databricks metadata, caches, or unrelated temporary
    files. Keeping this filter explicit makes the dataset explainable: when the
    notebook prints the source count, it is clear which kind of files were used.
    """
    ignored_parts = (".git", ".databricks", ".ipynb_checkpoints")
    if any(part in relative_path for part in ignored_parts):
        return False
    if not relative_path.endswith((".md", ".ipynb")):
        return False
    return bool(re.match(r"^(README\.md|[1-9]_.*)", relative_path))


def notebook_source_text(path: Path) -> str:
    """Extract the human-written parts of a notebook.

    A notebook file is JSON, not plain text. For language-model training we keep
    only markdown cells because they contain explanatory prose. Code cells are
    intentionally excluded; otherwise the decoder quickly learns Python syntax,
    paths, SDK calls, and notebook plumbing instead of workshop language.
    """
    notebook = json.loads(path.read_text(encoding="utf-8"))
    parts = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", [])
        source_text = "".join(source) if isinstance(source, list) else str(source)
        if not source_text.strip():
            continue
        if cell.get("cell_type") == "markdown":
            parts.append(source_text)
    return "\n".join(parts)


def read_source_text(path: Path) -> str:
    """Read one training source file in the format appropriate for its type.

    Markdown files can be read directly. Notebooks need JSON parsing first, so
    they go through `notebook_source_text`. The caller does not need to know
    those details; it just receives text that can be cleaned and tokenized.
    """
    if path.suffix == ".ipynb":
        return notebook_source_text(path)
    return path.read_text(encoding="utf-8")


def load_workshop_corpus(repo_root: Path) -> tuple[str, list[str]]:
    """Build one text corpus from the workshop repository.

    Language models are trained on long streams of text. For this demo we create
    that stream by concatenating markdown files and notebook markdown cells from
    the repository. The returned path list is logged later in MLflow so the training
    run remains reproducible and participants can inspect exactly what the model
    was allowed to learn from.
    """
    source_paths = [
        path
        for path in sorted(repo_root.rglob("*"))
        if is_workshop_source(path.relative_to(repo_root).as_posix())
    ]

    texts = []
    for path in source_paths:
        text = clean_text(read_source_text(path))
        if text:
            texts.append(text)

    if texts:
        return "\n".join(texts), [path.relative_to(repo_root).as_posix() for path in source_paths]

    raise ValueError(
        "No markdown or notebook prose was found for training. Check repo_root and workshop files."
    )


def tokenize(text: str) -> list[str]:
    """Split text into the small units seen by the model.

    Neural networks do not read Python strings directly. First, text is broken
    into tokens. This workshop tokenizer is deliberately simple: words become
    word tokens, and common punctuation marks become separate tokens. Production
    LLMs usually use subword tokenizers, but the rest of the pipeline is the
    same: tokens are the bridge between raw text and numbers.
    """
    return re.findall(TOKEN_PATTERN, text.lower())


def build_vocabulary(tokens: list[str], vocab_size: int) -> list[str]:
    """Create the token list that defines the model's known language.

    The vocabulary maps every known token to a stable integer id. We keep the
    most frequent tokens because a small model cannot reserve space for every
    rare word in the repository. Two special tokens are always placed first:
    `<pad>` fills empty context positions, and `<unk>` represents a token that
    was not included in the compact vocabulary.
    """
    token_counts = Counter(tokens)
    return SPECIAL_TOKENS + [token for token, _ in token_counts.most_common(vocab_size - 2)]


def tokens_to_ids(tokens: list[str], vocab: list[str]) -> np.ndarray:
    """Replace each token string with the integer id used by the neural network.

    Embedding layers work like lookup tables indexed by integers. This function
    performs that lookup for the whole corpus. If a token is missing from the
    vocabulary, it becomes `<unk>` instead of crashing, which is the same general
    idea used by many text systems when they meet an unseen word.
    """
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    return np.array([token_to_id.get(token, token_to_id["<unk>"]) for token in tokens], dtype=np.int32)


def build_examples(token_ids: np.ndarray, context_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Create input/target pairs for next-token prediction.

    A decoder-only language model learns from pairs such as:
    input = `machine learning`, target = `learning pipelines`. The target is
    the same sequence shifted one position to the left, so every position asks:
    "given the tokens up to here, what token should come next?" Fixed-size
    windows keep batching simple while still demonstrating the core LLM task.
    """
    sequence_length = context_length + 1
    windows = [
        token_ids[start : start + sequence_length]
        for start in range(0, len(token_ids) - sequence_length)
    ]
    windows = np.array(windows, dtype=np.int32)
    return windows[:, :-1], windows[:, 1:]


def decode_token_ids(ids: list[int] | np.ndarray, vocab: list[str]) -> str:
    """Convert generated token ids back into readable text.

    The model outputs ids, not words. This function reverses the vocabulary
    mapping and removes `<pad>` tokens, then fixes the most obvious spacing
    issues around punctuation. It is intentionally lightweight because the goal
    is to make generation inspectable rather than to build a production detokenizer.
    """
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    tokens = [id_to_token[int(token_id)] for token_id in ids if int(token_id) != token_to_id["<pad>"]]
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,:;!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text


def encode_prompt(prompt: str, vocab: list[str]) -> list[int]:
    """Prepare a user prompt for the decoder.

    Generation starts from text typed by a user. The prompt must go through the
    same tokenizer and vocabulary mapping as the training corpus; otherwise the
    model would receive ids with a different meaning than the ids it saw during
    training.
    """
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    return [token_to_id.get(token, token_to_id["<unk>"]) for token in tokenize(prompt)]


def pad_or_trim_context(ids: list[int], context_length: int, pad_token_id: int) -> list[int]:
    """Make a prompt fit the model's fixed context window.

    Transformer implementations usually process a bounded context window. If the
    prompt is shorter than that window, we add `<pad>` tokens on the left so the
    real prompt stays at the end. If it is longer, we keep the most recent tokens,
    because the next-token prediction should depend most on the latest context.
    """
    if len(ids) < context_length:
        return [pad_token_id] * (context_length - len(ids)) + ids
    return ids[-context_length:]


def sample_next_token(
    logits: np.ndarray,
    pad_token_id: int,
    unknown_token_id: int,
    temperature: float = 0.8,
    top_k: int = 20,
) -> int:
    """Choose the next generated token from the model's raw scores.

    The model returns logits: one unnormalized score per vocabulary token. We
    block `<pad>` and `<unk>` because they are useful internally but poor visible
    output. `top_k` limits sampling to the strongest candidates, and
    `temperature` controls how sharp or random the probability distribution is.
    """
    logits = logits.astype(np.float64)
    logits[pad_token_id] = -np.inf
    logits[unknown_token_id] = -np.inf
    top_k = min(max(int(top_k), 1), len(logits))
    top_ids = np.argpartition(logits, -top_k)[-top_k:]
    filtered = np.full_like(logits, -np.inf)
    filtered[top_ids] = logits[top_ids]
    scaled = filtered / max(float(temperature), 1e-6)
    probabilities = np.exp(scaled - np.nanmax(scaled))
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(np.arange(len(probabilities)), p=probabilities))


@tf.keras.utils.register_keras_serializable(package="Workshop")
class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    """Turn token ids into vectors that include content and position.

    A token id only says "which token is this?" The model also needs "where is
    this token in the context?" because the same word can play a different role
    at different positions. This layer learns both lookup tables and adds them
    together, producing one vector per token position.
    """

    def __init__(self, vocab_size: int, max_length: int, dim: int, **kwargs):
        """Create token and position lookup tables.

        `vocab_size` tells the token embedding how many distinct token ids can be
        looked up. `max_length` tells the position embedding how many context
        positions exist. `dim` is the vector size used by the rest of the
        transformer.
        """
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.dim = dim
        self.token_embedding = tf.keras.layers.Embedding(vocab_size, dim)
        self.position_embedding = tf.keras.layers.Embedding(max_length, dim)

    def call(self, token_batch):
        """Embed a batch shaped as `[batch, sequence_length]`.

        The token embedding is looked up for every id in the batch. The position
        embedding is looked up for positions `0..sequence_length-1` and is shared
        across all rows in the batch. Adding them gives the decoder the two
        pieces of information it needs before attention starts.
        """
        positions = tf.range(start=0, limit=tf.shape(token_batch)[-1], delta=1)
        return self.token_embedding(token_batch) + self.position_embedding(positions)

    def get_config(self):
        """Expose constructor values required by Keras serialization.

        MLflow may need to save and reload this custom layer. Keras can only do
        that reliably when it knows which arguments recreate the same layer, so
        every value passed to `__init__` is returned here.
        """
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "max_length": self.max_length,
                "dim": self.dim,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="Workshop")
class DecoderBlock(tf.keras.layers.Layer):
    """One transformer decoder block.

    The block has two stages. Causal self-attention lets every token combine
    information from earlier tokens in the same context. The feed-forward network
    then transforms each position independently. Residual connections and layer
    normalization keep optimization stable enough for a small workshop model.
    """

    def __init__(self, dim: int, heads: int, ffn_dim: int, dropout_rate: float, **kwargs):
        """Create attention, feed-forward, normalization, and dropout layers.

        `dim` is the representation width, `heads` splits attention into several
        independent views, and `ffn_dim` is the hidden width of the feed-forward
        network. Dropout is included so the demo model follows the same regular
        transformer pattern used at larger scale.
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.heads = heads
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout_rate
        self.self_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=heads,
            key_dim=dim // heads,
            dropout=dropout_rate,
        )
        self.attention_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.attention_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ffn_dim, activation="gelu"),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(dim),
            ]
        )
        self.ffn_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.ffn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False, return_attention=False):
        """Apply masked attention and the feed-forward network.

        `use_causal_mask=True` is the key decoder-only rule: position 5 can look
        at positions 0..5, but not at position 6 or later. Without that mask the
        model could cheat during training by seeing the future token it is
        supposed to predict.
        """
        attention_output, attention_scores = self.self_attention(
            x,
            x,
            use_causal_mask=True,
            return_attention_scores=True,
            training=training,
        )
        x = self.attention_norm(x + self.attention_dropout(attention_output, training=training))
        ffn_output = self.ffn(x, training=training)
        x = self.ffn_norm(x + self.ffn_dropout(ffn_output, training=training))
        if return_attention:
            return x, attention_scores
        return x

    def get_config(self):
        """Expose constructor values required by Keras serialization."""
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "heads": self.heads,
                "ffn_dim": self.ffn_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="Workshop")
class DecoderOnlyTransformer(tf.keras.Model):
    """Compact decoder-only language model.

    The model accepts a batch of token-id windows and returns logits for every
    position in every window. It is "decoder-only" because it has no separate
    encoder input; all information comes from the previous tokens in the same
    sequence. That is the architectural family used by GPT-style language
    models, only scaled down here so the mechanics stay visible.
    """

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        dim: int,
        heads: int,
        layers: int,
        ffn_dim: int,
        dropout_rate: float,
        **kwargs,
    ):
        """Assemble the full decoder-only model from workshop hyperparameters.

        The constructor wires together embedding, repeated decoder blocks, final
        normalization, and the language-model output head. The same arguments are
        saved on the object because `get_config` needs them when Keras or MLflow
        recreate the model.
        """
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.dim = dim
        self.heads = heads
        self.layers_count = layers
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout_rate
        self.embedding = TokenAndPositionEmbedding(vocab_size, max_length, dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.blocks = [DecoderBlock(dim, heads, ffn_dim, dropout_rate) for _ in range(layers)]
        self.final_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.lm_head = tf.keras.layers.Dense(vocab_size)

    def call(self, token_batch, training=False, return_attention=False):
        """Run a forward pass through embeddings, decoder blocks, and output head.

        The final dense layer produces one score for every token in the
        vocabulary at every context position. During training those scores are
        compared with the shifted target ids. During generation we use only the
        last position, because that is the model's prediction for the next token.
        """
        x = self.embedding(token_batch)
        x = self.dropout(x, training=training)
        first_block_attention = None
        for block_index, block in enumerate(self.blocks):
            if return_attention and block_index == 0:
                x, first_block_attention = block(x, training=training, return_attention=True)
            else:
                x = block(x, training=training)
        logits = self.lm_head(self.final_norm(x))
        if return_attention:
            return logits, first_block_attention
        return logits

    def get_config(self):
        """Expose constructor values required by Keras serialization."""
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "max_length": self.max_length,
                "dim": self.dim,
                "heads": self.heads,
                "layers": self.layers_count,
                "ffn_dim": self.ffn_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


def build_model(config: dict[str, Any]) -> DecoderOnlyTransformer:
    """Construct the decoder model from logged hyperparameters.

    The notebook uses the same configuration dictionary for training and model
    logging. Keeping construction in one function prevents drift between the
    architecture that is trained and the architecture registered in MLflow.
    """
    return DecoderOnlyTransformer(
        vocab_size=config["vocab_size"],
        max_length=config["context_length"],
        dim=config["embedding_dim"],
        heads=config["num_heads"],
        layers=config["num_layers"],
        ffn_dim=config["ffn_dim"],
        dropout_rate=config["dropout"],
    )


def generate_text(
    model: tf.keras.Model,
    prompt: str,
    vocab: list[str],
    context_length: int,
    steps: int = 60,
    temperature: float = 0.8,
    top_k: int = 20,
) -> str:
    """Generate text one token at a time.

    Autoregressive generation is a loop: encode the current text, keep the latest
    context window, ask the model for the next-token logits, sample one id, append
    it, and repeat. This is the same control flow used by much larger decoder
    language models, although production systems add stronger tokenizers,
    decoding strategies, safety filters, and serving infrastructure.
    """
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    generated_ids = encode_prompt(prompt, vocab)
    for _ in range(steps):
        context_ids = pad_or_trim_context(generated_ids, context_length, token_to_id["<pad>"])
        logits = model(np.array([context_ids], dtype=np.int32), training=False).numpy()[0, -1]
        generated_ids.append(
            sample_next_token(
                logits,
                pad_token_id=token_to_id["<pad>"],
                unknown_token_id=token_to_id["<unk>"],
                temperature=temperature,
                top_k=top_k,
            )
        )
    return decode_token_ids(generated_ids, vocab)
