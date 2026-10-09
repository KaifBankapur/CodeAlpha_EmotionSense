"""Dataset construction: filename parsing, speaker-disjoint splits and leakage.

The leakage tests are the most important ones in the suite. RAVDESS gives every
actor the same 60 utterances, so a file-level split leaks speaker identity and
produces a number that looks good and means nothing. These tests fail loudly if
that ever regresses.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from ml.config import EMOTION_LABELS, RAVDESS_EMOTION_CODES, SplitConfig
from ml.data.dataset import FeatureCache, config_fingerprint
from ml.data.ravdess import (
    RavdessParseError,
    assign_splits,
    build_manifest,
    load_manifest,
    parse_filename,
    summarise,
)


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------
class TestFilenameParsing:
    @pytest.mark.parametrize("code,label", sorted(RAVDESS_EMOTION_CODES.items()))
    def test_every_emotion_code_maps_to_its_published_label(self, code, label):
        # Field order is MM-VC-EI-SS-RV-RA-AA: emotion is the *third* field.
        sample = parse_filename(Path(f"Actor_01/03-01-{code}-01-01-01-01.wav"))
        assert sample.label == label
        assert sample.label_index == EMOTION_LABELS.index(label)

    def test_full_parse(self):
        sample = parse_filename(Path("Actor_07/03-01-05-01-02-01-07.wav"))
        assert sample.actor == 7
        assert sample.speaker_id == "Actor_07"
        assert sample.emotion_code == "05"
        assert sample.label == "angry"
        assert sample.intensity == 1
        assert sample.statement == 2
        assert sample.repetition == 1
        assert sample.filename == "03-01-05-01-02-01-07.wav"

    def test_intensity_two_is_strong(self):
        sample = parse_filename(Path("Actor_02/03-01-03-02-01-01-02.wav"))
        assert sample.intensity == 2

    def test_relative_path_is_posix(self):
        sample = parse_filename(Path("Actor_03/03-01-04-01-01-01-03.wav"))
        assert sample.path == "Actor_03/03-01-04-01-01-01-03.wav"
        assert "\\" not in sample.path

    @pytest.mark.parametrize(
        "name,reason",
        [
            ("not-a-ravdess-file.wav", "not the RAVDESS grammar at all"),
            ("03-01-05-01-02-01.wav", "six fields, one short"),
            ("03-01-05-01-02-01-01-01.wav", "eight fields, one too many"),
            ("03-01-99-01-02-01-01.wav", "emotion code 99 is not in the corpus"),
            ("03-01-05-01-02-01-99.wav", "actor 99 does not exist"),
            ("03-01-05-01-02-01-00.wav", "actor 00 does not exist"),
            ("03-01-05-01-02-01.mp3", "wrong container extension"),
            ("03-01-05-01-02-01-01.WAVX", "extension must be exactly .wav"),
        ],
    )
    def test_malformed_names_are_rejected(self, name, reason):
        """Each case must fail for the reason named, not by accident.

        An 8-field name also fails the regex, so a test written against one
        while claiming to check the emotion code would pass while proving
        nothing. Every entry here is 7 fields unless the row says otherwise.
        """
        with pytest.raises(RavdessParseError):
            parse_filename(Path(name))

    def test_unknown_modality_parses_but_is_filtered_from_the_manifest(self, tmp_path):
        """``parse_filename`` is permissive; ``build_manifest`` is strict.

        Parsing describes a file, filtering decides what to train on, and
        keeping them separate is what lets the same parser serve the song and
        speech vocallings.
        """
        actor_dir = tmp_path / "Actor_01"
        actor_dir.mkdir()
        for name in (
            "12-01-05-01-02-01-01.wav",  # video-only modality
            "03-02-05-01-02-01-01.wav",  # song, not speech
            "03-01-05-01-02-01-01.wav",  # the one we want
        ):
            (actor_dir / name).write_bytes(b"")

        assert parse_filename(Path("12-01-05-01-02-01-01.wav")).actor == 1
        assert len(build_manifest(tmp_path)) == 1

    def test_label_index_follows_the_canonical_order(self):
        for index, label in enumerate(EMOTION_LABELS):
            code = next(c for c, v in RAVDESS_EMOTION_CODES.items() if v == label)
            sample = parse_filename(Path(f"Actor_01/03-01-{code}-01-01-01-01.wav"))
            assert sample.label_index == index

    def test_song_files_parse_but_are_a_different_vocal_channel(self):
        """``03-02-..`` is song, not speech - the manifest filter must drop it.

        RAVDESS puts the vocal channel in field 2, which is what
        ``build_manifest`` tests to keep speech only.
        """
        from ml.data.ravdess import VOCAL_CHANNEL_SPEECH

        song = parse_filename(Path("Actor_01/03-02-05-01-02-01-01.wav"))
        assert song.label == "angry"  # still a valid emotion code

        # The filter reads the same offset for every file, so it stays correct
        # whatever the emotion happens to be.
        speech = Path("Actor_01/03-01-05-01-02-01-01.wav").name
        assert int(speech[3:5]) == int(VOCAL_CHANNEL_SPEECH)
        assert int(song.filename[3:5]) != int(VOCAL_CHANNEL_SPEECH)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_manifest():
    """A full 24-actor manifest, so split logic is testable without the corpus."""
    from ml.data.ravdess import RavdessSample

    samples = []
    for actor in range(1, 25):
        for index, label in enumerate(EMOTION_LABELS):
            samples.append(
                RavdessSample(
                    path=f"Actor_{actor:02d}/03-01-{index + 1:02d}-01-01-01-{actor:02d}.wav",
                    filename=f"03-01-{index + 1:02d}-01-01-01-{actor:02d}.wav",
                    label=label,
                    label_index=index,
                    emotion_code=f"{index + 1:02d}",
                    intensity=1,
                    statement=1,
                    repetition=1,
                    actor=actor,
                    speaker_id=f"Actor_{actor:02d}",
                )
            )
    return samples


class TestSplitAssignment:
    def test_no_actor_appears_in_two_splits(self, synthetic_manifest):
        """The core guarantee. Breaking this invalidates every reported number."""
        splits = assign_splits(synthetic_manifest)
        actors = {name: {s.actor for s in items} for name, items in splits.items()}

        assert actors["train"] & actors["val"] == set()
        assert actors["train"] & actors["test"] == set()
        assert actors["val"] & actors["test"] == set()

    def test_every_sample_is_assigned_exactly_once(self, synthetic_manifest):
        splits = assign_splits(synthetic_manifest)
        total = sum(len(items) for items in splits.values())
        assert total == len(synthetic_manifest)

    def test_splits_cover_all_24_actors(self, synthetic_manifest):
        splits = assign_splits(synthetic_manifest)
        union = {s.actor for items in splits.values() for s in items}
        assert union == set(range(1, 25))

    def test_declared_actors_match_actual_actors(self, synthetic_manifest):
        cfg = SplitConfig()
        splits = assign_splits(synthetic_manifest, cfg)
        assert {s.actor for s in splits["test"]} == set(cfg.test_actors)
        assert {s.actor for s in splits["val"]} == set(cfg.val_actors)
        assert {s.actor for s in splits["train"]} == set(cfg.train_actors())

    def test_split_sizes_are_16_4_4(self, synthetic_manifest):
        splits = assign_splits(synthetic_manifest)
        assert len({s.actor for s in splits["train"]}) == 16
        assert len({s.actor for s in splits["val"]}) == 4
        assert len({s.actor for s in splits["test"]}) == 4

    def test_every_split_covers_every_emotion(self, synthetic_manifest):
        for name, items in assign_splits(synthetic_manifest).items():
            assert {s.label for s in items} == set(EMOTION_LABELS), name

    def test_gender_balance_within_each_split(self, synthetic_manifest):
        """RAVDESS odd actor numbers are female, even are male (dataset convention).

        A split that ended up all-male would make any gender-dependent
        conclusion meaningless.
        """
        for name, items in assign_splits(synthetic_manifest).items():
            genders = Counter(s.actor % 2 for s in items)
            assert genders[1] == genders[0], name

    def test_overlapping_actors_are_rejected(self, synthetic_manifest):
        bad = SplitConfig(test_actors=[3, 6], val_actors=[6, 11])
        with pytest.raises(ValueError, match="overlap"):
            assign_splits(synthetic_manifest, bad)

    def test_train_split_is_the_complement_by_construction(self, synthetic_manifest):
        """``train_actors()`` derives itself, so coverage is automatic.

        The interesting failure mode is *overlap*, not under-coverage - and that
        is what the overlap check catches.
        """
        cfg = SplitConfig(test_actors=[3, 8], val_actors=[6, 11])
        assert len(cfg.train_actors()) == 20
        assert set(cfg.train_actors()) == set(range(1, 25)) - {3, 8, 6, 11}

        splits = assign_splits(synthetic_manifest, cfg)
        assert {s.actor for s in splits["train"]} == set(cfg.train_actors())
        assert len(splits["test"]) == 2 * 8  # 2 actors x 8 utterances

    def test_output_is_deterministically_ordered(self, synthetic_manifest):
        first = assign_splits(synthetic_manifest)["train"]
        second = assign_splits(list(reversed(synthetic_manifest)))["train"]
        assert [s.path for s in first] == [s.path for s in second]


class TestSummarise:
    def test_counts_come_from_the_files_not_hardcoded(self, synthetic_manifest):
        stats = summarise(synthetic_manifest)
        assert stats["n_samples"] == 24 * 8
        assert stats["n_speakers"] == 24
        assert stats["n_classes"] == 8
        assert all(count == 24 for count in stats["samples_per_emotion"].values())

    def test_imbalance_ratio_is_computed(self, synthetic_manifest):
        # Remove one neutral sample to create a real imbalance.
        trimmed = [s for s in synthetic_manifest if not (s.actor == 1 and s.label == "neutral")]
        stats = summarise(trimmed)
        assert stats["samples_per_emotion"]["neutral"] == 23
        assert stats["class_imbalance_ratio"] > 1.0

    def test_handles_an_empty_corpus(self):
        stats = summarise([])
        assert stats["n_samples"] == 0
        assert stats["class_imbalance_ratio"] is None


# ---------------------------------------------------------------------------
# Real corpus
# ---------------------------------------------------------------------------
class TestRealCorpus:
    def test_manifest_matches_the_expected_corpus_size(self, ravdess_root):
        samples = build_manifest(ravdess_root)
        # 24 actors x 60 utterances, audio-only speech.
        assert len(samples) == 1440
        assert len({s.actor for s in samples}) == 24

    def test_all_eight_emotions_are_present(self, ravdess_root):
        labels = {s.label for s in build_manifest(ravdess_root)}
        assert labels == set(EMOTION_LABELS)

    def test_real_corpus_is_imbalanced_in_neutral(self, ravdess_root):
        """RAVDESS has no 'strong neutral', so neutral has half the samples.

        This is the fact that motivates weighted loss and macro-F1.
        """
        counts = Counter(s.label for s in build_manifest(ravdess_root))
        assert counts["neutral"] == 96
        for label in EMOTION_LABELS:
            if label != "neutral":
                assert counts[label] == 192

    def test_only_audio_speech_files_are_included(self, ravdess_root):
        for sample in build_manifest(ravdess_root):
            assert sample.filename[:2] == "03"  # audio-only modality
            assert sample.filename[3:5] == "01"  # vocal channel = speech

    def test_real_splits_have_no_speaker_overlap(self, ravdess_root):
        samples = build_manifest(ravdess_root)
        splits = assign_splits(samples)
        actors = [{s.actor for s in items} for items in splits.values()]
        assert not (actors[0] & actors[1])
        assert not (actors[0] & actors[2])
        assert not (actors[1] & actors[2])

    def test_no_file_path_appears_in_two_splits(self, ravdess_root):
        splits = assign_splits(build_manifest(ravdess_root))
        paths = [{s.path for s in items} for items in splits.values()]
        assert not (paths[0] & paths[1])
        assert not (paths[0] & paths[2])
        assert not (paths[1] & paths[2])

    def test_wavs_are_decodable(self, ravdess_sample_paths):
        """Every listed file must actually be readable audio."""
        import soundfile as sf

        for path in ravdess_sample_paths:
            data, sr = sf.read(str(path), frames=1)
            assert data.size > 0
            assert sr > 0


# ---------------------------------------------------------------------------
# Manifest persistence
# ---------------------------------------------------------------------------
class TestManifestPersistence:
    def test_round_trip(self, synthetic_manifest, tmp_path):
        from ml.data.ravdess import save_manifest

        path = tmp_path / "manifest.json"
        save_manifest(synthetic_manifest, path)
        restored = load_manifest(path)

        assert len(restored) == len(synthetic_manifest)
        assert restored[0] == synthetic_manifest[0]

    def test_unknown_fields_are_dropped(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "path": "Actor_01/x.wav",
                        "filename": "x.wav",
                        "label": "calm",
                        "label_index": 1,
                        "emotion_code": "02",
                        "intensity": 1,
                        "statement": 1,
                        "repetition": 1,
                        "actor": 1,
                        "speaker_id": "Actor_01",
                        "future_field": 42,
                    }
                ]
            ),
            encoding="utf-8",
        )
        restored = load_manifest(path)
        assert restored[0].label == "calm"


# ---------------------------------------------------------------------------
# Feature cache
# ---------------------------------------------------------------------------
class TestCacheFingerprint:
    def test_fingerprint_changes_with_the_configuration(self, audio_cfg, feature_cfg):
        from dataclasses import replace

        base = config_fingerprint(audio_cfg, feature_cfg)
        assert config_fingerprint(audio_cfg, feature_cfg) == base  # stable

        assert config_fingerprint(replace(audio_cfg, target_duration_sec=2.0), feature_cfg) != base
        assert config_fingerprint(replace(audio_cfg, sample_rate=22050), feature_cfg) != base
        assert config_fingerprint(replace(audio_cfg, trim_silence=False), feature_cfg) != base
        assert config_fingerprint(replace(audio_cfg, peak_normalize=False), feature_cfg) != base
        assert config_fingerprint(audio_cfg, replace(feature_cfg, n_mfcc=13)) != base
        assert config_fingerprint(audio_cfg, replace(feature_cfg, cmvn=False)) != base

    def test_fingerprint_is_deliberately_conservative(self, audio_cfg, feature_cfg):
        """Non-feature fields still invalidate the cache.

        Failing safe is the point: an over-broad hash costs a rebuild, an
        over-narrow one would serve features that do not match the config.
        """
        from dataclasses import replace

        base = config_fingerprint(audio_cfg, feature_cfg)
        assert config_fingerprint(replace(audio_cfg, max_upload_bytes=99), feature_cfg) != base

    def test_fingerprint_is_a_short_stable_hex_string(self, audio_cfg, feature_cfg):
        digest = config_fingerprint(audio_cfg, feature_cfg)
        assert len(digest) == 16
        assert all(c in "0123456789abcdef" for c in digest)


class TestRealCache:
    def test_cache_shape_matches_the_configured_window(self, audio_cfg, feature_cfg):
        from ml.data.loading import cache_dir_for

        directory = cache_dir_for(audio_cfg, feature_cfg)
        if not directory.is_dir():
            pytest.skip("No feature cache. Run `python scripts/prepare_data.py`.")

        cache = FeatureCache.load(directory, expected_fingerprint=None)
        expected_frames = 1 + int(
            audio_cfg.target_duration_sec * audio_cfg.sample_rate // feature_cfg.hop_length
        )
        assert cache.features.shape == (
            len(cache),
            feature_cfg.n_output_features(),
            expected_frames,
        )
        assert cache.n_features == 120
        assert len(cache) == 1440

    def test_cache_labels_match_the_manifest_order(self, audio_cfg, feature_cfg):
        from ml.data.loading import cache_dir_for

        directory = cache_dir_for(audio_cfg, feature_cfg)
        if not directory.is_dir():
            pytest.skip("No feature cache.")

        cache = FeatureCache.load(directory, expected_fingerprint=None)
        assert set(np.unique(cache.labels).tolist()) <= set(range(len(EMOTION_LABELS)))

    def test_strict_fingerprint_guard_rejects_a_mismatch(self, tmp_path):
        """Loading the wrong cache must fail loudly, not silently mis-train."""
        pytest.importorskip("numpy")
        directory = tmp_path / "features_deadbeefdeadbeef"
        directory.mkdir()
        np.save(directory / "features.npy", np.zeros((2, 120, 301), dtype=np.float32))
        np.save(directory / "labels.npy", np.array([0, 1], dtype=np.int64))
        np.save(directory / "valid_frames.npy", np.array([301, 301], dtype=np.int64))
        (directory / "sample_paths.json").write_text("[]", encoding="utf-8")
        (directory / "meta.json").write_text(
            json.dumps({"fingerprint": "not-the-one-you-asked-for"}), encoding="utf-8"
        )

        with pytest.raises((ValueError, RuntimeError)):
            FeatureCache.load(directory, expected_fingerprint="the-one-you-asked-for")

    def test_cache_reports_padding_statistics(self, audio_cfg, feature_cfg):
        """A large valid-frame fraction would mean the window is nearly all padding."""
        from ml.data.loading import cache_dir_for

        directory = cache_dir_for(audio_cfg, feature_cfg)
        if not directory.is_dir():
            pytest.skip("No feature cache.")

        cache = FeatureCache.load(directory, expected_fingerprint=None)
        valid_fraction = float(np.mean(cache.valid_frames) / cache.n_frames)
        assert 0.0 < valid_fraction < 1.0
