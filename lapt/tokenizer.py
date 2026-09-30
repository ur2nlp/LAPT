"""Train a new tokenizer and settle where its special tokens land.

`TokenizerArtifact` is the cached entry point: it trains a SentencePiece model
over the target-language corpus, converts it to a HuggingFace backend through
`lapt_core.spm`, and re-establishes the base checkpoint's special tokens on the
result.

That last part is the subtle one. Which ids the special tokens end up on is a
decision, not a detail -- a copied post-processor and a stored generation config
both address them by id. The policy implemented here mints them into
SentencePiece at their original ids where that is possible and reassigns them
positionally where it is not; see the published guide on special-token policies
for the alternative and when a base model needs it.

Initializing the new vocabulary's *embeddings* is `lapt.focus`.
"""

import json
import os
import sys
from collections.abc import Mapping
from typing import Any

import sentencepiece as spm
from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast

from lapt.artifact_configs import TokenizerConfig
from lapt_core.artifacts import ArtifactConfig, CachedArtifact
from lapt_core.spm import create_bpe_backend, create_unigram_backend



class TokenizerArtifact(CachedArtifact):
    """A FOCUS tokenizer trained by SentencePiece, cached on `TokenizerConfig`.

    Not used for the `focus.tokenizer_path` bypass (a pre-built tokenizer, e.g.
    PTEx): that case is handled entirely by the caller, which loads it directly
    and never constructs this artifact. `tokenizer_config.tokenizer_path` is
    therefore always falsy here.

    `path` is `{root}/{language}/{tokenizer_id}` rather than the `root/name`
    default, since the tokenizer id already encodes every parameter that
    should distinguish one cache from another. This reproduces
    `TokenizerConfig.cache_dir(language)` exactly at the default `root`
    ("tokenizers"), which production callers rely on; the parameter exists so
    tests can point a whole cache tree at `tmp_path` instead.

    `config_filename` is `training_config.yaml`, matching the name this
    artifact's caching logic already wrote before the port -- unlike the
    source layer, no migration is needed for existing caches to be read.

    `artifact_config()` is overridden to return `tokenizer_config` itself
    rather than the default dict-wrapping: `TokenizerConfig.check_cached`
    already strips embedding-only and retired seed-vocabulary fields before
    diffing (see `lapt/artifact_configs.py`), and that tolerance would be
    silently lost if validation went through the generic `_DictArtifactConfig`
    path instead.

    Behavior change from the pre-port code: a cache directory that exists but
    carries no `training_config.yaml` at all now raises
    `MissingConfigRecordError` instead of being silently accepted with a
    warning. Same tightening already applied to the untokenized-source layer
    (see architecture.md's "Landed" notes) -- there is nothing to distinguish
    an untracked pre-existing cache from one interrupted mid-write, so both
    must be refused.
    """

    name = "tokenizer"
    depends_on = ("untokenized",)
    config_filename = "training_config.yaml"

    def __init__(
        self,
        language: str,
        tokenizer_config: TokenizerConfig,
        jsonl_path: str | None = None,
        root: str = "tokenizers",
    ):
        """Initialize the artifact.

        Args:
            language: Language code, used only for the cache path.
            tokenizer_config: Parameters the tokenizer is trained and cached
                with.
            jsonl_path: Path to the FOCUS training-data JSONL, required only
                when the cache turns out to be cold. Callers that can tell in
                advance that the cache is warm (see `exists()`) may skip
                preparing this and construct without it.
            root: Directory the per-language tokenizer caches live under.
                Defaults to the production convention; override in tests.
        """
        super().__init__(root=root)
        self.language = language
        self.tokenizer_config = tokenizer_config
        self.jsonl_path = jsonl_path

    @property
    def path(self) -> str:
        return os.path.join(self.root, self.language, self.tokenizer_config.tokenizer_id())

    def config(self) -> dict:
        return self.tokenizer_config.to_dict()

    def artifact_config(self) -> ArtifactConfig:
        return self.tokenizer_config

    def build(self, deps: Mapping[str, Any]) -> PreTrainedTokenizerFast:
        """Train a new tokenizer on `self.jsonl_path` using SentencePiece.

        The tokenizer will use the same algorithm (BPE, Unigram, etc.) as the
        base tokenizer.

        Writes SentencePiece's own model files (`spm.model`, `spm.vocab`)
        directly into `self.path` as a side effect of training -- HF's
        SentencePiece-to-fast-tokenizer conversion needs `spm.model` to exist
        on disk, so unlike a builder that returns a self-contained value, this
        one cannot defer all filesystem writes to `write()`. `self.path` is
        therefore created here rather than left to `resolve()`'s later
        `os.makedirs`, which runs after `build()` returns.
        """
        if self.jsonl_path is None:
            raise ValueError(
                f"No cached tokenizer at {self.path}, but jsonl_path was not "
                "provided to train one. Prepare the FOCUS training data first "
                "and pass it to TokenizerArtifact."
            )

        config = self.tokenizer_config
        output_path = self.path
        os.makedirs(output_path, exist_ok=True)

        print(f"Training new tokenizer with vocab size {config.vocab_size}", file=sys.stderr)

        # Inspect base tokenizer to determine algorithm and special tokens to inherit
        # Force Fast tokenizer since we need to access backend_tokenizer for algorithm detection
        base_tokenizer = AutoTokenizer.from_pretrained(config.hf_model, use_fast=True)

        detected_type = _detect_tokenizer_algorithm(base_tokenizer)
        if config.tokenizer_algorithm is not None:
            model_type = config.tokenizer_algorithm
            if model_type == detected_type:
                print(f"Tokenizer algorithm: {model_type} (explicitly set, matches base)", file=sys.stderr)
            else:
                print(
                    f"Tokenizer algorithm: {model_type} "
                    f"(explicitly set, overrides base's {detected_type})",
                    file=sys.stderr,
                )
        else:
            model_type = detected_type
            print(f"Tokenizer algorithm: {model_type} (inherited from base)", file=sys.stderr)

        special_tokens_config = _extract_special_tokens(
            base_tokenizer,
            inherit_additional=config.inherit_additional_special_tokens,
            vocab_size=config.vocab_size,
        )

        # Convert JSONL to plain text for SentencePiece training (cached alongside JSONL)
        # We keep the JSONL for FOCUS which needs that format later
        text_file_path = self.jsonl_path.replace('.jsonl', '_spm.txt')
        if not os.path.exists(text_file_path):
            print(f"Creating SentencePiece training file: {text_file_path}", file=sys.stderr)
            with open(self.jsonl_path, encoding='utf-8') as jsonl_file:
                with open(text_file_path, 'w', encoding='utf-8') as text_file:
                    for line in jsonl_file:
                        data = json.loads(line)
                        text_file.write(data['text'] + '\n')
        else:
            print(f"SentencePiece training file already exists: {text_file_path}", file=sys.stderr)

        sp_model = _train_sentencepiece_model(
            text_file_path=text_file_path,
            model_type=model_type,
            vocab_size=config.vocab_size,
            special_tokens_config=special_tokens_config,
            output_path=output_path,
            character_coverage=config.character_coverage,
        )

        # Extract vocabulary with scores for HuggingFace tokenizer initialization
        actual_vocab_size = sp_model.get_piece_size()
        vocab_with_scores = [
            (sp_model.id_to_piece(i), sp_model.get_score(i))
            for i in range(actual_vocab_size)
        ]

        # Convert SentencePiece model to HuggingFace tokenizer backend. Both branches
        # build the model manually and apply the same SentencePiece pipeline via
        # lapt_core.spm, so Unigram and BPE stay as comparable as possible.
        if model_type == 'bpe':
            model_file = os.path.join(output_path, 'spm.model')
            backend_tokenizer = create_bpe_backend(
                spm_model_path=model_file,
                vocab_scores=vocab_with_scores,
                unk_token=special_tokens_config['unk_piece'],
            )
        else:
            backend_tokenizer = create_unigram_backend(
                vocab_with_scores,
                unk_id=special_tokens_config['unk_id'],
            )

        _copy_base_post_processor(backend_tokenizer, base_tokenizer, special_tokens_config)

        # Wrap in PreTrainedTokenizerFast with special tokens resolved against the
        # trained vocabulary rather than read straight off the base tokenizer, whose
        # roles may have been renamed, aliased, or synthesized during training.
        trained_vocab = {piece for piece, _score in vocab_with_scores}
        hf_special_tokens = _resolve_hf_special_tokens(
            base_tokenizer,
            special_tokens_config,
            trained_vocab,
        )
        print(f"Registering special tokens on the new tokenizer: {hf_special_tokens}", file=sys.stderr)
        new_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend_tokenizer,
            bos_token=hf_special_tokens['bos_token'],
            eos_token=hf_special_tokens['eos_token'],
            unk_token=hf_special_tokens['unk_token'],
            pad_token=hf_special_tokens['pad_token'],
            clean_up_tokenization_spaces=True,
        )

        # Add additional special tokens ONLY if we inherited them
        # (They're already in the SentencePiece vocab via user_defined_symbols,
        #  but PreTrainedTokenizerFast needs to know about them explicitly.
        #  This is REGISTRATION not ADDITION - we're just setting the
        #  additional_special_tokens attribute, not increasing vocab size)
        if config.inherit_additional_special_tokens:
            if hasattr(base_tokenizer, 'additional_special_tokens') and base_tokenizer.additional_special_tokens:
                new_tokenizer.add_special_tokens({
                    'additional_special_tokens': base_tokenizer.additional_special_tokens
                })

        _validate_special_token_ids(new_tokenizer, special_tokens_config)
        _validate_tokenizer(new_tokenizer, config.vocab_size)
        return new_tokenizer

    def write(self, value: PreTrainedTokenizerFast, path: str) -> None:
        value.save_pretrained(path)

    def read(self, path: str) -> PreTrainedTokenizerFast:
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        _validate_tokenizer(tokenizer, self.tokenizer_config.vocab_size)
        return tokenizer


def _train_sentencepiece_model(
    text_file_path: str,
    model_type: str,
    vocab_size: int,
    special_tokens_config: dict,
    output_path: str,
    character_coverage: float = 1.0,
) -> spm.SentencePieceProcessor:
    """
    Train a SentencePiece model and return the loaded processor.

    Args:
        text_file_path: Path to plain text training file (one sentence per line)
        model_type: 'bpe' or 'unigram'
        vocab_size: Target vocabulary size
        special_tokens_config: Dict of special token configs (from _extract_special_tokens)
        output_path: Directory where model files will be saved
        character_coverage: Fraction of character occurrences to cover (0-1)

    Returns:
        Loaded SentencePieceProcessor with the trained model
    """
    model_prefix = os.path.join(output_path, 'spm')

    # Train SentencePiece model
    # character_coverage: fraction of character occurrences to cover (rest become UNK)
    # normalization_rule_name='identity': no text normalization
    # hard_vocab_limit=True: strictly enforce vocab_size (not a soft target)
    train_args = {
        'input': text_file_path,
        'model_prefix': model_prefix,
        'model_type': model_type,
        'vocab_size': vocab_size,
        'character_coverage': character_coverage,
        'normalization_rule_name': 'identity',
        'hard_vocab_limit': True,
    }
    train_args.update(special_tokens_config)

    # Pass args via the keyword API rather than a space-joined command-line
    # string: user_defined_symbols is a list (see _extract_special_tokens) whose
    # literal '\n' piece would otherwise be mangled by string arg parsing.
    print(f"Training SentencePiece with args: {train_args}", file=sys.stderr)

    spm.SentencePieceTrainer.Train(**train_args)

    # Load the trained model and validate vocab size
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(f'{model_prefix}.model')

    actual_vocab_size = sp_model.get_piece_size()
    print(f"SentencePiece model trained. Vocab size: {actual_vocab_size}", file=sys.stderr)

    if actual_vocab_size != vocab_size:
        raise ValueError(
            f"Trained SentencePiece model has vocab size {actual_vocab_size}, "
            f"but expected {vocab_size}. This may indicate a SentencePiece training issue."
        )

    return sp_model


def _detect_tokenizer_algorithm(tokenizer: PreTrainedTokenizerFast) -> str:
    """
    Detect whether a tokenizer uses BPE or Unigram algorithm.

    Requires a Fast tokenizer (PreTrainedTokenizerFast) to access backend_tokenizer.

    Args:
        tokenizer: HuggingFace Fast tokenizer to inspect

    Returns:
        'bpe' or 'unigram'
    """
    backend_model = tokenizer.backend_tokenizer.model
    model_type_str = str(type(backend_model).__name__).lower()

    if 'bpe' in model_type_str:
        return 'bpe'
    elif 'unigram' in model_type_str:
        return 'unigram'
    else:
        raise ValueError(
            f"Unknown tokenizer algorithm: {type(backend_model)}. "
            "Expected BPE or Unigram."
        )





def _copy_base_post_processor(
    backend_tokenizer,
    base_tokenizer: PreTrainedTokenizerBase,
    special_tokens_config: dict,
) -> None:
    """
    Copy the base tokenizer's post-processor onto the new backend, when portable.

    Only a ``TemplateProcessing`` post-processor is portable, and only if the base
    special tokens kept their ids. It is defined over special-token strings and
    their ids — XGLM's prepends ``</s>`` to every input, which the adapted model
    still expects. Other post-processors belong to their own pipeline: Qwen3's
    ``ByteLevel`` post-processor assumes byte-level pre-tokenization and would
    corrupt offsets on the metaspace pipeline ``lapt_core.spm`` installs.

    Args:
        backend_tokenizer: Newly built ``tokenizers.Tokenizer`` to modify in place
        base_tokenizer: Base tokenizer to copy from
        special_tokens_config: Output of _extract_special_tokens
    """
    from tokenizers.processors import TemplateProcessing

    base_post_processor = getattr(base_tokenizer, '_tokenizer', None)
    if base_post_processor is not None:
        base_post_processor = getattr(base_post_processor, 'post_processor', None)

    if base_post_processor is None:
        return

    if not isinstance(base_post_processor, TemplateProcessing):
        print(
            f"Skipping base post-processor ({type(base_post_processor).__name__}): only "
            "TemplateProcessing is portable across tokenization pipelines",
            file=sys.stderr,
        )
        return

    if not _base_special_token_ids_preserved(base_tokenizer, special_tokens_config):
        print(
            "Skipping base TemplateProcessing post-processor: special-token ids were "
            "reassigned during training, so the template's hard-coded ids no longer match",
            file=sys.stderr,
        )
        return

    backend_tokenizer.post_processor = base_post_processor
    print(
        "Copied post-processor from base tokenizer (preserves special token handling)",
        file=sys.stderr,
    )


def _validate_tokenizer(tokenizer: PreTrainedTokenizerBase, expected_vocab_size: int):
    """
    Validate that a tokenizer has the expected vocab size and contiguous token IDs.

    Args:
        tokenizer: Tokenizer to validate
        expected_vocab_size: Expected vocabulary size

    Raises:
        ValueError: If validation fails
    """
    actual_vocab_size = len(tokenizer)

    if actual_vocab_size != expected_vocab_size:
        raise ValueError(
            f"Tokenizer has vocab size {actual_vocab_size}, "
            f"but expected {expected_vocab_size}"
        )

    # Check that token IDs are contiguous from 0 to vocab_size-1
    # HuggingFace's .train_new_from_iterator() had a bug where it would skip ID 0
    # at larger vocab sizes (e.g., creating IDs 1-4095 instead of 0-4095 for vocab_size=4096)
    vocab = tokenizer.get_vocab()
    all_token_ids = list(vocab.values())

    if len(all_token_ids) != actual_vocab_size:
        raise ValueError(
            f"Vocab has {len(all_token_ids)} entries but vocab_size is {actual_vocab_size}"
        )

    min_id = min(all_token_ids)
    max_id = max(all_token_ids)

    if min_id != 0 or max_id != actual_vocab_size - 1:
        raise ValueError(
            f"Token IDs are not contiguous! Range is {min_id}-{max_id}, "
            f"expected 0-{actual_vocab_size - 1}"
        )

    # Check for duplicates or gaps in token IDs
    unique_ids = set(all_token_ids)
    if len(unique_ids) != actual_vocab_size:
        raise ValueError(
            f"Token IDs have duplicates or gaps! "
            f"Found {len(unique_ids)} unique IDs but expected {actual_vocab_size}"
        )

    print(f"Tokenizer validation passed: vocab_size={actual_vocab_size}, token IDs: {min_id}-{max_id}", file=sys.stderr)


SPECIAL_TOKEN_ROLES = ('unk', 'bos', 'eos', 'pad')
DEFAULT_UNK_PIECE = '<unk>'


def _assign_special_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    role_pieces: dict[str, str],
    vocab_size: int | None,
) -> dict[str, int]:
    """
    Choose SentencePiece ids for the special-token roles that have a piece.

    The base model's own ids are preserved whenever SentencePiece can accept them,
    so tokenizers built against bases like XGLM (unk/bos/eos/pad at 3/0/2/1) are
    unchanged. They are unusable when a role has no id, when two roles share an
    id, or when an id falls outside the target vocabulary — the last case is the
    norm for a large-vocab base such as Qwen3, whose eos id is 151643. Then ids
    are reassigned positionally from 0 in SPECIAL_TOKEN_ROLES order.

    Args:
        tokenizer: Base tokenizer to read ids from
        role_pieces: Mapping of role name to the piece string it owns
        vocab_size: Target vocabulary size, or None to skip the range check

    Returns:
        Mapping of role name to SentencePiece id, covering exactly role_pieces
    """
    base_ids = {}
    for role in role_pieces:
        base_id = getattr(tokenizer, f'{role}_token_id')
        if base_id is not None:
            base_ids[role] = base_id

    base_ids_usable = (
        len(base_ids) == len(role_pieces)
        and len(set(base_ids.values())) == len(base_ids)
        and all(base_id >= 0 for base_id in base_ids.values())
        and (vocab_size is None or all(base_id < vocab_size for base_id in base_ids.values()))
    )
    if base_ids_usable:
        return base_ids

    assigned_ids = {}
    next_id = 0
    for role in SPECIAL_TOKEN_ROLES:
        if role in role_pieces:
            assigned_ids[role] = next_id
            next_id += 1

    print(
        "Base special-token ids are not usable for SentencePiece "
        f"(base: {base_ids}, vocab_size: {vocab_size}); "
        f"reassigning positionally: {assigned_ids}",
        file=sys.stderr,
    )
    return assigned_ids


def _validate_special_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    special_tokens_config: dict,
) -> None:
    """
    Check that each special-token role landed on the id it was assigned.

    The ids in ``special_tokens_config`` are a *request*: they are handed to
    SentencePiece as ``unk_id``/``bos_id``/``eos_id``/``pad_id`` and then have to
    survive the conversion to a HuggingFace backend and the role resolution in
    ``_resolve_hf_special_tokens``, which drops a role to None when its piece is
    absent from the trained vocabulary. Nothing downstream re-checks the result,
    so a role that silently went missing surfaces much later and indirectly: a
    vocabulary-adapted model whose eos is gone never emits a stop token, and
    generation runs to max_new_tokens on every prompt.

    Roles the base model lacks carry id -1 and are skipped, as is an aliased role
    such as Qwen3's pad, which is deliberately trained with -1 and recovered by
    string.

    Args:
        tokenizer: The tokenizer just built
        special_tokens_config: Output of _extract_special_tokens

    Raises:
        ValueError: If a requested role is missing or sits on a different id
    """
    mismatches = []
    for role in SPECIAL_TOKEN_ROLES:
        requested_id = special_tokens_config.get(f'{role}_id', -1)
        if requested_id < 0:
            continue
        actual_id = getattr(tokenizer, f'{role}_token_id', None)
        if actual_id != requested_id:
            piece = special_tokens_config.get(f'{role}_piece')
            mismatches.append(
                f"{role} ({piece!r}): requested id {requested_id}, got {actual_id}"
            )

    if mismatches:
        raise ValueError(
            "Special-token ids did not survive tokenizer construction: "
            + "; ".join(mismatches)
        )


def _base_special_token_ids_preserved(
    tokenizer: PreTrainedTokenizerBase,
    special_tokens_config: dict,
) -> bool:
    """
    Report whether every special token the base model has kept its original id.

    Used to decide whether artifacts that hard-code base ids — notably a
    ``TemplateProcessing`` post-processor — can be copied onto the new tokenizer.

    Args:
        tokenizer: Base tokenizer
        special_tokens_config: Output of _extract_special_tokens

    Returns:
        True if no base special token was reassigned to a different id
    """
    for role in SPECIAL_TOKEN_ROLES:
        base_id = getattr(tokenizer, f'{role}_token_id')
        if base_id is None:
            continue
        if special_tokens_config.get(f'{role}_id') != base_id:
            return False
    return True


def _resolve_hf_special_tokens(
    tokenizer: PreTrainedTokenizerBase,
    special_tokens_config: dict,
    vocab: set[str],
) -> dict[str, str | None]:
    """
    Choose the special-token strings to register on the new ``PreTrainedTokenizerFast``.

    A role resolves to the piece SentencePiece was told to mint for it, falling
    back to the base tokenizer's string. The fallback is what re-attaches an
    aliased role: Qwen3 sets ``pad_token == eos_token``, which SentencePiece
    cannot accept twice, so ``pad`` is trained with id -1 and recovered here as
    the same string — HuggingFace then resolves it to the eos id. A role whose
    string is absent from the trained vocabulary resolves to None rather than
    silently registering a token that would be added to the vocab.

    Args:
        tokenizer: Base tokenizer
        special_tokens_config: Output of _extract_special_tokens
        vocab: Piece strings present in the trained SentencePiece model

    Returns:
        Mapping of ``<role>_token`` to a piece string or None
    """
    resolved = {}
    for role in SPECIAL_TOKEN_ROLES:
        piece = special_tokens_config.get(f'{role}_piece')
        if piece is None:
            piece = getattr(tokenizer, f'{role}_token')
        resolved[f'{role}_token'] = piece if piece in vocab else None
    return resolved


def _extract_special_tokens(
    tokenizer: PreTrainedTokenizerBase,
    inherit_additional: bool = True,
    vocab_size: int | None = None,
) -> dict:
    """
    Extract special token configuration from a tokenizer for SentencePiece training.

    Args:
        tokenizer: HuggingFace tokenizer to extract special tokens from
        inherit_additional: Whether to inherit additional special tokens (e.g., <madeupword0-6>)
            from the base tokenizer (default: True)
        vocab_size: Target vocabulary size. Used only to check that the base
            model's special-token ids fit; pass it whenever it is known, or
            SentencePiece training will fail on a large-vocab base.

    Returns:
        Dictionary of SentencePiece training arguments for special tokens
    """
    config = {}

    # Always reserve a piece for the newline character. SentencePiece treats the
    # training file as one sentence per line and strips '\n' as the line
    # delimiter, so a newline piece never emerges from the corpus on its own.
    # Modern LMs are universally expected to handle newlines (chat templates,
    # multi-line documents, code), so inject it unconditionally as a
    # user-defined symbol rather than gating it behind a config flag.
    user_defined_symbols = ['\n']

    # Work out which role owns which piece string. A base tokenizer may alias two
    # roles to one string (Qwen3 sets pad_token == eos_token), but SentencePiece
    # rejects a duplicate meta piece outright. The earlier role in
    # SPECIAL_TOKEN_ROLES order keeps the string; the later one is disabled here
    # and re-attached to the HuggingFace wrapper by _resolve_hf_special_tokens.
    claimed_pieces = set()
    role_pieces = {}
    for role in SPECIAL_TOKEN_ROLES:
        piece = getattr(tokenizer, f'{role}_token')
        if piece is None or piece in claimed_pieces:
            continue
        claimed_pieces.add(piece)
        role_pieces[role] = piece

    # SentencePiece always mints an unknown piece and both backend models index it
    # by id, so synthesize one when the base has none. Byte-level BPE bases such
    # as Qwen3 cannot produce UNK at all and expose unk_token=None, but the
    # non-byte-level tokenizer trained here does need a real unknown piece.
    if 'unk' not in role_pieces and DEFAULT_UNK_PIECE not in claimed_pieces:
        role_pieces['unk'] = DEFAULT_UNK_PIECE
        claimed_pieces.add(DEFAULT_UNK_PIECE)

    role_ids = _assign_special_token_ids(tokenizer, role_pieces, vocab_size)

    for role in SPECIAL_TOKEN_ROLES:
        if role in role_pieces:
            config[f'{role}_piece'] = role_pieces[role]
        # Emit an id for every role, including -1 for roles the base model lacks.
        # Leaving one out lets SentencePiece apply its own defaults (<s> at id 1,
        # </s> at id 2), which would either invent a special token the base model
        # has no notion of or collide with a reassigned id.
        config[f'{role}_id'] = role_ids.get(role, -1)

    # Optionally inherit additional special tokens like <madeupword0-6>
    # These are vocabulary reservations from the base model that may be unused
    if inherit_additional:
        if hasattr(tokenizer, 'additional_special_tokens') and tokenizer.additional_special_tokens:
            # dedupe against symbols already reserved (e.g. the newline piece)
            for token in tokenizer.additional_special_tokens:
                if token not in user_defined_symbols:
                    user_defined_symbols.append(token)

    # Return as a list (not a comma-joined string) so SentencePiece receives each
    # symbol intact via the kwargs API — required for the literal '\n' piece,
    # which cannot survive a space-joined command-line argument string.
    config['user_defined_symbols'] = user_defined_symbols

    return config
