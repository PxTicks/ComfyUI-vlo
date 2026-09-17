"""vloSaveVideo: per-frame resize during native-style encoding."""

from __future__ import annotations

import json
from fractions import Fraction

import av
import numpy as np
import pytest
import torch

from test_batch_nodes_integration import nodes_module  # noqa: F401


@pytest.fixture
def video_save(nodes_module, tmp_path, monkeypatch):  # noqa: F811
    import folder_paths

    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path / "output"))
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path / "temp"))
    module = nodes_module.video_save
    node = module.vloSaveVideo
    monkeypatch.setattr(
        node,
        "hidden",
        nodes_module.io.HiddenHolder.from_dict({"PROMPT": {"1": {"class_type": "x"}}}),
        raising=False,
    )
    return module


def _frames(count=4, height=36, width=64):
    # Distinct flat colours per frame so decoded output can be checked loosely.
    images = torch.zeros(count, height, width, 3)
    for index in range(count):
        images[index, :, :, index % 3] = 0.8
    return images


def _execute(module, images, **overrides):
    kwargs = dict(
        fps=24.0,
        filename_prefix="video/test",
        save_output=True,
        width=0,
        height=0,
        upscale_method="bicubic",
        crop="disabled",
        format="auto",
        codec="auto",
        crf=18.0,
    )
    kwargs.update(overrides)
    return module.vloSaveVideo.execute(images, **kwargs)


def _saved_path(result, tmp_path):
    item = result.ui.as_dict()["images"][0]
    return item, tmp_path / item["type"] / item["subfolder"] / item["filename"]


def test_resolve_output_size_follows_image_scale_with_even_sides(video_save) -> None:
    resolve = video_save.resolve_output_size
    assert resolve(64, 36, 0, 0) == (64, 36)
    assert resolve(1920, 1080, 3840, 2160) == (3840, 2160)
    assert resolve(1920, 1080, 0, 720) == (1280, 720)
    # 854.2 -> 854; an odd derived side is rounded to even.
    assert resolve(1920, 1080, 0, 480) == (854, 480)
    assert resolve(1000, 999, 0, 100) == (100, 100)
    with pytest.raises(ValueError, match="even"):
        resolve(64, 36, 65, 36)
    with pytest.raises(ValueError, match="even"):
        resolve(63, 36, 0, 0)


def test_local_video_configuration_resolves_native_choices(video_save) -> None:
    options, container, codec = video_save.video_output_config("output", "auto", "auto")
    assert container == video_save.Types.VideoContainer.MP4
    assert codec == video_save.Types.VideoCodec.H264
    assert options == {
        "mode": "w",
        "format": "mp4",
        "options": {"movflags": "use_metadata_tags+faststart"},
    }

    options, container, codec = video_save.video_output_config(
        "output.webm", "auto", "auto"
    )
    assert container == video_save.Types.VideoContainer.WEBM
    assert codec == video_save.Types.VideoCodec.AV1
    assert options == {"mode": "w", "format": "webm"}
    assert video_save.video_encoder_options(codec, 0) == {
        "svtav1-params": "lossless=1"
    }


def test_center_crop_box_matches_common_upscale(video_save) -> None:
    import comfy.utils

    source = torch.arange(1 * 3 * 36 * 64, dtype=torch.float32).reshape(1, 3, 36, 64)
    for width, height in ((36, 36), (64, 16), (100, 100)):
        x, y, crop_width, crop_height = video_save.center_crop_box(64, 36, width, height)
        cropped = source[:, :, y:y + crop_height, x:x + crop_width]
        assert torch.equal(
            comfy.utils.common_upscale(source, width, height, "bilinear", "center"),
            torch.nn.functional.interpolate(cropped, size=(height, width), mode="bilinear"),
        )


def test_saves_resized_mp4_to_output_with_metadata(video_save, tmp_path) -> None:
    images = _frames()
    result = _execute(video_save, images, width=128, height=72)

    item, path = _saved_path(result, tmp_path)
    assert item["type"] == "output"
    assert item["subfolder"] == "video"
    assert path.name.endswith(".mp4") and path.exists()

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == (128, 72)
        assert stream.average_rate == Fraction(24)
        assert stream.format.name == "yuv420p"
        assert json.loads(container.metadata["prompt"]) == {"1": {"class_type": "x"}}
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    assert len(frames) == 4
    # Frame 1 is green: the dominant channel survives resize + encode.
    assert frames[1][36, 64].argmax() == 1

    video = result.result[0]
    assert video.get_dimensions() == (128, 72)


def test_scaled_8bit_video_preserves_sub_byte_source_detail(video_save, tmp_path) -> None:
    images = torch.empty(2, 4, 4, 3)
    images[0].fill_(1.05 / 255)
    images[1].fill_(1.95 / 255)
    assert torch.equal((images[0] * 255).byte(), (images[1] * 255).byte())

    result = _execute(video_save, images, width=8, height=8, crf=0)
    _, path = _saved_path(result, tmp_path)
    with av.open(str(path)) as container:
        frames = [frame.to_ndarray(format="yuv420p") for frame in container.decode(video=0)]

    assert len(frames) == 2
    assert not np.array_equal(frames[0], frames[1])


def test_save_output_false_previews_from_temp(video_save, tmp_path) -> None:
    result = _execute(video_save, _frames(), save_output=False, width=32)

    item, path = _saved_path(result, tmp_path)
    assert item["type"] == "temp"
    assert path.exists()
    assert not (tmp_path / "output").exists()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == (32, 18)


def test_unscaled_save_matches_native_save_to(video_save, nodes_module, tmp_path) -> None:  # noqa: F811
    images = torch.rand(3, 36, 64, 3)
    result = _execute(video_save, images, crf=23.0)
    _, path = _saved_path(result, tmp_path)

    native_path = tmp_path / "native.mp4"
    nodes_module.InputImpl.VideoFromComponents(
        nodes_module.Types.VideoComponents(images=images, frame_rate=Fraction(24)),
    ).save_to(str(native_path), crf=23.0)

    def decode(file):
        with av.open(str(file)) as container:
            return [frame.to_ndarray(format="yuv420p") for frame in container.decode(video=0)]

    ours, native = decode(path), decode(native_path)
    assert len(ours) == len(native) == 3
    for a, b in zip(ours, native):
        np.testing.assert_array_equal(a, b)


def test_center_crop_resize_keeps_the_middle(video_save, tmp_path) -> None:
    images = torch.zeros(2, 36, 64, 3)
    images[:, :, :14, 0] = 1.0  # red bands on the sides, cropped away
    images[:, :, 50:, 0] = 1.0
    images[:, :, 14:50, 2] = 1.0  # blue centre
    result = _execute(video_save, images, width=72, height=72, crop="center")

    _, path = _saved_path(result, tmp_path)
    with av.open(str(path)) as container:
        frame = next(container.decode(video=0)).to_ndarray(format="rgb24")
    assert frame.shape == (72, 72, 3)
    assert frame[36, 4].argmax() == 2
    assert frame[36, 68].argmax() == 2


def test_webm_av1_10bit_hdr_with_resampled_audio(video_save, tmp_path) -> None:
    sample_rate = 44100
    audio = {"waveform": torch.zeros(1, 2, sample_rate), "sample_rate": sample_rate}
    result = _execute(
        video_save,
        _frames(count=6),
        format="webm",
        codec="auto",
        width=64,
        height=36,
        crf=40.0,
        audio=audio,
        color_space="HDR",
    )

    _, path = _saved_path(result, tmp_path)
    assert path.suffix == ".webm"
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        assert video.codec_context.name in {"libdav1d", "libaom-av1", "av1"}
        assert video.format.name == "yuv420p10le"
        assert video.color_trc == av.video.reformatter.ColorTrc.ARIB_STD_B67
        audio_stream = container.streams.audio[0]
        assert audio_stream.codec_context.name in {"opus", "libopus"}
        assert audio_stream.sample_rate == 48000


def test_rejects_bad_codec_settings(video_save) -> None:
    with pytest.raises(ValueError, match="crf"):
        _execute(video_save, _frames(), codec="h264", crf=60.0)
    with pytest.raises(ValueError, match="AV1"):
        _execute(video_save, _frames(), format="webm", codec="h264")


def test_interrupt_removes_partial_file(video_save, monkeypatch, tmp_path) -> None:
    import comfy.model_management

    calls = {"count": 0}

    def interrupt_on_third_frame():
        calls["count"] += 1
        if calls["count"] == 3:
            raise comfy.model_management.InterruptProcessingException()

    monkeypatch.setattr(
        comfy.model_management,
        "throw_exception_if_processing_interrupted",
        interrupt_on_third_frame,
    )
    with pytest.raises(comfy.model_management.InterruptProcessingException):
        _execute(video_save, _frames(count=5), width=128, height=72)
    assert list((tmp_path / "output" / "video").iterdir()) == []
