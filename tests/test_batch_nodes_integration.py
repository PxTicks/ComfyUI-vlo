from __future__ import annotations

import importlib.util
import io
import os
import sys
import types
from fractions import Fraction
from pathlib import Path

import pytest
import torch
from PIL import Image


PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def _load_nodes_module():
    raw_comfyui_path = os.environ.get("COMFYUI_PATH")
    if not raw_comfyui_path:
        pytest.skip("Set COMFYUI_PATH to run ComfyUI node integration tests")

    comfyui_path = Path(raw_comfyui_path).resolve()
    if not (comfyui_path / "comfy_api").is_dir():
        pytest.fail(f"COMFYUI_PATH is not a ComfyUI checkout: {comfyui_path}")

    sys.path.insert(0, str(comfyui_path))
    original_argv = sys.argv
    original_server = sys.modules.get("server")
    try:
        sys.argv = [original_argv[0], "--cpu"]
        import comfy.options

        comfy.options.enable_args_parsing()

        class Routes:
            def get(self, _path):
                return lambda function: function

            def post(self, _path):
                return lambda function: function

            def delete(self, _path):
                return lambda function: function

        class PromptServerInstance:
            routes = Routes()
            client_id = None

            def send_sync(self, *_args, **_kwargs):
                return None

        class PromptServer:
            instance = PromptServerInstance()

        server = types.ModuleType("server")
        server.PromptServer = PromptServer
        sys.modules["server"] = server

        package_name = "comfyui_vlo_batch_test"
        package = types.ModuleType(package_name)
        package.__path__ = [str(PLUGIN_ROOT)]
        sys.modules[package_name] = package

        spec = importlib.util.spec_from_file_location(
            f"{package_name}.nodes",
            PLUGIN_ROOT / "nodes.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load ComfyUI-vlo nodes module")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = original_argv
        if original_server is None:
            sys.modules.pop("server", None)
        else:
            sys.modules["server"] = original_server


@pytest.fixture(scope="module")
def nodes_module():
    return _load_nodes_module()


def _register_png(nodes_module, *, size: tuple[int, int], name: str) -> str:
    payload = io.BytesIO()
    Image.new("RGB", size, (30, 60, 90)).save(payload, format="PNG")
    return nodes_module.REGISTRY.register(
        kind="image",
        filename=name,
        content_type="image/png",
        data=payload.getvalue(),
    ).media_id


def test_batch_node_schemas_publish_comfy_list_outputs(nodes_module) -> None:
    expected = (
        (nodes_module.vloMemoryLoadImageBatch, ["IMAGE", "MASK"], [True, True]),
        (nodes_module.vloMemoryLoadAudioBatch, ["AUDIO"], [True]),
        (nodes_module.vloMemoryLoadVideoBatch, ["VIDEO", "BOOLEAN"], [True, True]),
    )

    for node_class, return_types, output_is_list in expected:
        schema = node_class.GET_SCHEMA()
        assert list(node_class.RETURN_TYPES) == return_types
        assert list(node_class.OUTPUT_IS_LIST) == output_is_list
        assert schema.inputs[0].as_dict()["options"] == []
        assert "remote" not in schema.inputs[0].as_dict()


def test_image_batch_validator_checks_empty_unknown_and_wrong_kind(nodes_module) -> None:
    image_id = _register_png(nodes_module, size=(8, 6), name="valid.png")
    audio_id = nodes_module.REGISTRY.register(
        kind="audio",
        filename="wrong.wav",
        content_type="audio/wav",
        data=b"not decoded during validation",
    ).media_id

    node_class = nodes_module.vloMemoryLoadImageBatch
    assert node_class.validate_inputs([image_id]) is True
    assert node_class.validate_inputs([]) == "Select at least one image"
    assert node_class.validate_inputs(["missing-id"]) == "Invalid image id: missing-id"
    assert "expected 'image'" in node_class.validate_inputs([audio_id])


def test_image_batch_executes_independent_image_sizes_in_order(nodes_module) -> None:
    first_id = _register_png(nodes_module, size=(8, 6), name="first.png")
    second_id = _register_png(nodes_module, size=(5, 9), name="second.png")

    images, masks = nodes_module.vloMemoryLoadImageBatch.execute(
        [first_id, second_id]
    ).result

    assert [tuple(image.shape) for image in images] == [
        (1, 6, 8, 3),
        (1, 9, 5, 3),
    ]
    assert len(masks) == 2


def test_video_batch_emits_audio_flags_aligned_with_its_videos(
    nodes_module,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        nodes_module.InputImpl,
        "VideoFromFile",
        lambda source: ("video", source),
    )
    media_ids = [
        nodes_module.REGISTRY.register(
            kind="video",
            filename=f"clip-{index}.mp4",
            content_type="video/mp4",
            data=f"clip-{index}".encode(),
        ).media_id
        for index in range(3)
    ]

    node_class = nodes_module.vloMemoryLoadVideoBatch
    videos, audio_flags = node_class.execute(
        media_ids, include_audio="0,1"
    ).result

    assert len(videos) == 3
    # One flag per video, defaulting the item the user never toggled.
    assert audio_flags == [False, True, False]
    assert node_class.execute(media_ids).result[1] == [False, False, False]


def test_video_batch_reports_flag_problems_and_fingerprints_them(
    nodes_module,
) -> None:
    media_id = nodes_module.REGISTRY.register(
        kind="video",
        filename="clip.mp4",
        content_type="video/mp4",
        data=b"clip",
    ).media_id

    node_class = nodes_module.vloMemoryLoadVideoBatch
    assert node_class.validate_inputs([media_id], include_audio="1") is True
    assert (
        node_class.validate_inputs([media_id], include_audio="1,1")
        == "Video audio inclusion has 2 flags for 1 items"
    )

    # Flipping a flag has to invalidate the cached execution.
    assert node_class.fingerprint_inputs(
        [media_id], include_audio="1"
    ) != node_class.fingerprint_inputs([media_id], include_audio="0")


def test_minimax_batch_adapter_schema_accepts_comfy_lists(nodes_module) -> None:
    node_class = nodes_module.vloMiniMaxH3ReferenceToVideoBatch
    schema = node_class.GET_SCHEMA()

    assert node_class.INPUT_IS_LIST is True
    io_types = {input_spec.id: input_spec.io_type for input_spec in schema.inputs}
    assert {
        name: io_types.get(name)
        for name in ("ref_images", "ref_videos", "ref_video_audios", "ref_audios")
    } == {
        "ref_images": "IMAGE",
        "ref_videos": "VIDEO",
        "ref_video_audios": "AUDIO",
        "ref_audios": "AUDIO",
    }
    # Per-video audio gating rides a BOOLEAN input so a single widget value and a
    # connected per-video list both work without a schema change.
    assert io_types.get("use_embedded_video_audio") == "BOOLEAN"
    assert schema.enable_expand is True


def _fake_native_minimax_node(
    nodes_module,
    *,
    image_max: int = 9,
    video_max: int = 3,
    video_audio_max: int = 3,
    audio_max: int = 3,
):
    class FakeNativeMiniMaxNode:
        @classmethod
        def GET_SCHEMA(cls):
            return nodes_module.io.Schema(
                node_id="MiniMaxH3ReferenceToVideo",
                inputs=[
                    nodes_module.io.Clip.Input("clip"),
                    nodes_module.io.Vae.Input("vae"),
                    nodes_module.io.Vae.Input("audio_vae"),
                    nodes_module.io.String.Input("prompt"),
                    nodes_module.io.Int.Input("width"),
                    nodes_module.io.Int.Input("height"),
                    nodes_module.io.Int.Input("length"),
                    nodes_module.io.Combo.Input(
                        "ref_image_size",
                        options=["match", "max"],
                    ),
                    nodes_module.io.Autogrow.Input(
                        "ref_images",
                        optional=True,
                        template=nodes_module.io.Autogrow.TemplatePrefix(
                            input=nodes_module.io.Image.Input("ref_image"),
                            prefix="native_picture_",
                            min=0,
                            max=image_max,
                        ),
                    ),
                    nodes_module.io.Autogrow.Input(
                        "ref_videos",
                        optional=True,
                        template=nodes_module.io.Autogrow.TemplatePrefix(
                            input=nodes_module.io.Image.Input("ref_video"),
                            prefix="native_video_",
                            min=0,
                            max=video_max,
                        ),
                    ),
                    nodes_module.io.Autogrow.Input(
                        "ref_video_audios",
                        optional=True,
                        template=nodes_module.io.Autogrow.TemplatePrefix(
                            input=nodes_module.io.Audio.Input("ref_video_audio"),
                            prefix="native_soundtrack_",
                            min=0,
                            max=video_audio_max,
                        ),
                    ),
                    nodes_module.io.Autogrow.Input(
                        "ref_audios",
                        optional=True,
                        template=nodes_module.io.Autogrow.TemplatePrefix(
                            input=nodes_module.io.Audio.Input("ref_audio"),
                            prefix="native_audio_",
                            min=0,
                            max=audio_max,
                        ),
                    ),
                ],
                outputs=[
                    nodes_module.io.Conditioning.Output(),
                    nodes_module.io.Latent.Output(),
                ],
            )

    return FakeNativeMiniMaxNode


def _expanded_native_node(result):
    assert result.expand is not None
    assert len(result.expand) == 1
    node_id, node = next(iter(result.expand.items()))
    assert result.result == ([node_id, 0], [node_id, 1])
    return node


def _finalize_expanded_native_inputs(native_node, expanded):
    # Exercise the same V3 dynamic-input finalization that ComfyUI runs before
    # invoking the expanded node. Autogrow members are only recognized when the
    # GraphBuilder key includes the parent input id (for example,
    # `ref_audios.ref_audio_0`).
    from comfy_api.latest import _io

    raw_inputs = expanded["inputs"]
    class_inputs = _io.create_input_dict_v1(native_node.GET_SCHEMA().inputs)
    finalized, _, v3_data = _io.get_finalized_class_inputs(
        class_inputs,
        raw_inputs,
    )
    recognized_ids = set(finalized.get("required", {})) | set(
        finalized.get("optional", {})
    )
    execution_inputs = {
        input_id: value
        for input_id, value in raw_inputs.items()
        if input_id in recognized_ids
    }
    return _io.build_nested_inputs(execution_inputs, v3_data)


def test_minimax_batch_adapter_expands_ordered_native_inputs(
    nodes_module,
    monkeypatch,
) -> None:
    embedded_audio = {"waveform": torch.zeros(1, 2, 10), "sample_rate": 32000}
    override_audio = {"waveform": torch.ones(1, 2, 10), "sample_rate": 32000}
    standalone_audio = {"waveform": torch.full((1, 2, 10), 2.0), "sample_rate": 32000}

    class FakeVideo:
        def get_components(self):
            return types.SimpleNamespace(
                images=torch.arange(8 * 2 * 3 * 3, dtype=torch.float32).reshape(
                    8, 2, 3, 3
                ),
                frame_rate=Fraction(12, 1),
                audio=embedded_audio,
            )

    native_node = _fake_native_minimax_node(nodes_module)
    monkeypatch.setattr(
        nodes_module,
        "_get_native_minimax_h3_reference_node",
        lambda: native_node,
    )

    first_image = torch.zeros(1, 4, 5, 3)
    second_image = torch.ones(1, 6, 7, 3)
    result = nodes_module.vloMiniMaxH3ReferenceToVideoBatch.execute(
        clip=["clip"],
        vae=["vae"],
        audio_vae=["audio-vae"],
        prompt=["<Picture 1> and <Video 1>"],
        width=[1344],
        height=[768],
        length=[124],
        ref_image_size=["match"],
        ref_images=[first_image, second_image],
        ref_videos=[FakeVideo()],
        ref_video_audios=[override_audio],
        ref_audios=[standalone_audio],
    )

    expanded = _expanded_native_node(result)
    assert expanded["class_type"] == "MiniMaxH3ReferenceToVideo"
    native_inputs = expanded["inputs"]
    assert native_inputs["ref_images.native_picture_0"] is first_image
    assert native_inputs["ref_images.native_picture_1"] is second_image
    assert native_inputs["ref_videos.native_video_0"].shape[0] == 16
    assert (
        native_inputs["ref_video_audios.native_soundtrack_0"] is override_audio
    )
    assert native_inputs["ref_audios.native_audio_0"] is standalone_audio
    assert native_inputs["prompt"] == "<Picture 1> and <Video 1>"
    assert "ref_images.native_picture_2" not in native_inputs

    finalized_inputs = _finalize_expanded_native_inputs(native_node, expanded)
    assert set(finalized_inputs["ref_images"]) == {
        "native_picture_0",
        "native_picture_1",
    }
    assert finalized_inputs["ref_images"]["native_picture_0"] is first_image
    assert finalized_inputs["ref_images"]["native_picture_1"] is second_image
    assert finalized_inputs["ref_videos"]["native_video_0"].shape[0] == 16
    assert (
        finalized_inputs["ref_video_audios"]["native_soundtrack_0"]
        is override_audio
    )
    assert finalized_inputs["ref_audios"]["native_audio_0"] is standalone_audio


def test_minimax_batch_adapter_uses_embedded_audio_and_schema_limits(
    nodes_module,
    monkeypatch,
) -> None:
    embedded_audio = {"waveform": torch.zeros(1, 2, 10), "sample_rate": 32000}

    class FakeVideo:
        def get_components(self):
            return types.SimpleNamespace(
                images=torch.zeros(5, 2, 3, 3),
                frame_rate=Fraction(24, 1),
                audio=embedded_audio,
            )

    monkeypatch.setattr(
        nodes_module,
        "_get_native_minimax_h3_reference_node",
        lambda: _fake_native_minimax_node(nodes_module, image_max=2),
    )
    base_args = {
        "clip": ["clip"],
        "vae": ["vae"],
        "audio_vae": ["audio-vae"],
        "prompt": ["prompt"],
        "width": [1344],
        "height": [768],
        "length": [124],
        "ref_image_size": ["match"],
    }

    result = nodes_module.vloMiniMaxH3ReferenceToVideoBatch.execute(
        **base_args,
        ref_videos=[FakeVideo()],
        use_embedded_video_audio=[True],
    )
    expanded = _expanded_native_node(result)
    assert (
        expanded["inputs"]["ref_video_audios.native_soundtrack_0"]
        is embedded_audio
    )

    with pytest.raises(ValueError, match="at most 2 reference images"):
        nodes_module.vloMiniMaxH3ReferenceToVideoBatch.execute(
            **base_args,
            ref_images=[torch.zeros(1, 1, 1, 3)] * 3,
        )

    with pytest.raises(ValueError, match="cannot outnumber reference videos"):
        nodes_module.vloMiniMaxH3ReferenceToVideoBatch.execute(
            **base_args,
            ref_video_audios=[embedded_audio],
        )

    monkeypatch.setattr(
        nodes_module,
        "_get_native_minimax_h3_reference_node",
        lambda: _fake_native_minimax_node(
            nodes_module,
            video_max=2,
            video_audio_max=1,
        ),
    )
    with pytest.raises(ValueError, match="at most 1 reference video soundtrack"):
        nodes_module.vloMiniMaxH3ReferenceToVideoBatch.execute(
            **base_args,
            ref_videos=[FakeVideo(), FakeVideo()],
            use_embedded_video_audio=[True],
        )


def test_minimax_batch_adapter_gates_embedded_audio_per_video(
    nodes_module,
    monkeypatch,
) -> None:
    first_audio = {"waveform": torch.zeros(1, 2, 10), "sample_rate": 32000}
    second_audio = {"waveform": torch.ones(1, 2, 10), "sample_rate": 32000}
    override_audio = {"waveform": torch.full((1, 2, 10), 2.0), "sample_rate": 32000}

    def fake_video(audio):
        class FakeVideo:
            def get_components(self):
                return types.SimpleNamespace(
                    images=torch.zeros(5, 2, 3, 3),
                    frame_rate=Fraction(24, 1),
                    audio=audio,
                )

        return FakeVideo()

    monkeypatch.setattr(
        nodes_module,
        "_get_native_minimax_h3_reference_node",
        lambda: _fake_native_minimax_node(nodes_module),
    )
    base_args = {
        "clip": ["clip"],
        "vae": ["vae"],
        "audio_vae": ["audio-vae"],
        "prompt": ["prompt"],
        "width": [1344],
        "height": [768],
        "length": [124],
        "ref_image_size": ["match"],
    }

    def soundtracks(**kwargs):
        result = nodes_module.vloMiniMaxH3ReferenceToVideoBatch.execute(
            **base_args, **kwargs
        )
        inputs = _expanded_native_node(result)["inputs"]
        return {
            name: value
            for name, value in inputs.items()
            if name.startswith("ref_video_audios.native_soundtrack_")
        }

    videos = [fake_video(first_audio), fake_video(second_audio)]

    # Off by default: a reference video's own sound is not an <Audio N> reference.
    assert soundtracks(ref_videos=videos) == {}

    # A single widget value broadcasts to every video.
    assert soundtracks(ref_videos=videos, use_embedded_video_audio=[True]) == {
        "ref_video_audios.native_soundtrack_0": first_audio,
        "ref_video_audios.native_soundtrack_1": second_audio,
    }

    # A per-video list binds positionally, leaving a hole for the disabled video.
    assert soundtracks(
        ref_videos=videos, use_embedded_video_audio=[False, True]
    ) == {"ref_video_audios.native_soundtrack_1": second_audio}

    # An explicit override still wins for a video with embedded audio disabled.
    assert soundtracks(
        ref_videos=videos,
        ref_video_audios=[override_audio],
        use_embedded_video_audio=[False, False],
    ) == {"ref_video_audios.native_soundtrack_0": override_audio}

    with pytest.raises(ValueError, match="one value per reference video"):
        nodes_module.vloMiniMaxH3ReferenceToVideoBatch.execute(
            **base_args,
            ref_videos=videos,
            use_embedded_video_audio=[True, False, True],
        )


def test_minimax_batch_adapter_rejects_native_schema_drift(
    nodes_module,
    monkeypatch,
) -> None:
    native_node = _fake_native_minimax_node(nodes_module)
    original_get_schema = native_node.GET_SCHEMA

    @classmethod
    def incompatible_schema(cls):
        schema = original_get_schema()
        ref_videos = next(
            input_spec
            for input_spec in schema.inputs
            if input_spec.id == "ref_videos"
        )
        ref_videos.template.input = nodes_module.io.Video.Input("ref_video")
        return schema

    native_node.GET_SCHEMA = incompatible_schema
    monkeypatch.setattr(
        nodes_module,
        "_get_native_minimax_h3_reference_node",
        lambda: native_node,
    )

    with pytest.raises(
        RuntimeError,
        match="ref_videos.*expected IMAGE, got VIDEO",
    ):
        nodes_module._get_native_minimax_h3_reference_contract()
