from __future__ import annotations

import importlib.util
import io
import os
import sys
import types
from pathlib import Path

import pytest
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
        (nodes_module.vloMemoryLoadVideoBatch, ["VIDEO"], [True]),
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
