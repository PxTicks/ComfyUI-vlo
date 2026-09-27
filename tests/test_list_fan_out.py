from __future__ import annotations

import asyncio
import uuid

import pytest
import torch

from test_batch_nodes_integration import nodes_module  # noqa: F401


SLOT_COUNT = 16


def _image_list_source(io):
    class TestImageList(io.ComfyNode):
        """Stands in for vloMemoryLoadImageBatch: an IMAGE output list."""

        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="TestFanOutImageList",
                inputs=[io.Int.Input("count", default=2, min=0, max=64)],
                outputs=[io.Image.Output(is_output_list=True)],
            )

        @classmethod
        def execute(cls, count):
            # Mixed sizes, as the batch loader allows.
            return io.NodeOutput(
                [torch.full((1, 8 + i, 16, 3), float(i)) for i in range(count)]
            )

    return TestImageList


def _autogrow_sink(io, seen):
    class TestAutogrowSink(io.ComfyNode):
        """Same input group and None handling as TextEncodeQwenImage21."""

        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="TestFanOutAutogrowSink",
                is_output_node=True,
                inputs=[
                    io.Autogrow.Input(
                        "images",
                        template=io.Autogrow.TemplateNames(
                            io.Image.Input("image"),
                            names=[f"image_{i}" for i in range(1, 17)],
                            min=0,
                        ),
                    ),
                ],
                outputs=[],
            )

        @classmethod
        def execute(cls, images=None):
            images = images or {}
            used = []
            for name in sorted(images, key=lambda n: int(n.rsplit("_", 1)[-1])):
                if images[name] is not None:
                    used.append((name, tuple(images[name].shape)))
            seen.append(used)
            return io.NodeOutput()

    return TestAutogrowSink


class _Server:
    client_id = None
    last_node_id = None
    last_prompt_id = None

    def send_sync(self, *_args, **_kwargs):
        return None


def _flattened_subgraph_prompt(count, *, wired=10, loader_bypassed=False):
    # graphToPrompt flattens the subgraph, so its inner encode node links
    # straight to the fan-out's sockets; a bypassed loader drops the link.
    prompt = {
        "2": {"class_type": "vloListFanOut", "inputs": {}},
        "459:474": {
            "class_type": "TestFanOutAutogrowSink",
            "inputs": {f"images.image_{i + 1}": ["2", i] for i in range(wired)},
        },
    }
    if not loader_bypassed:
        prompt["1"] = {"class_type": "TestFanOutImageList", "inputs": {"count": count}}
        prompt["2"]["inputs"]["values"] = ["1", 0]
    return prompt


@pytest.fixture
def run_prompt(nodes_module, monkeypatch):  # noqa: F811
    import execution
    import nodes

    seen: list = []
    for node_class in (
        _image_list_source(nodes_module.io),
        _autogrow_sink(nodes_module.io, seen),
        nodes_module.vloListFanOut,
    ):
        monkeypatch.setitem(
            nodes.NODE_CLASS_MAPPINGS, node_class.GET_SCHEMA().node_id, node_class
        )

    def run(prompt):
        seen.clear()

        async def go():
            prompt_id = str(uuid.uuid4())
            valid = await execution.validate_prompt(prompt_id, prompt, None)
            assert valid[0], valid[1]
            executor = execution.PromptExecutor(
                _Server(),
                cache_type=execution.CacheType.CLASSIC,
                cache_args={"lru": 0, "ram": 0, "ram_inactive": 0},
            )
            await executor.execute_async(prompt, prompt_id, {}, valid[2])
            errors = [
                data.get("exception_message")
                for event, data in executor.status_messages
                if event == "execution_error"
            ]
            return executor.success, errors

        success, errors = asyncio.run(go())
        return success, list(seen), errors

    return run


def test_fan_out_outputs_share_the_input_template(nodes_module) -> None:  # noqa: F811
    io = nodes_module.io
    schema = nodes_module.vloListFanOut.GET_SCHEMA()

    assert schema.is_input_list is True
    (values,) = schema.inputs
    assert isinstance(values, io.MatchType.Input)
    assert values.optional is True
    assert len(schema.outputs) == SLOT_COUNT
    assert all(isinstance(output, io.MatchType.Output) for output in schema.outputs)
    assert {output.template.template_id for output in schema.outputs} == {
        values.template.template_id
    }
    assert not any(output.is_output_list for output in schema.outputs)


def test_fan_out_pads_short_lists_with_none(run_prompt) -> None:
    success, seen, errors = run_prompt(_flattened_subgraph_prompt(2))

    assert success, errors
    # One sink execution: the fan-out's outputs are single values, not lists.
    assert seen == [[("image_1", (1, 8, 16, 3)), ("image_2", (1, 9, 16, 3))]]


def test_fan_out_fills_every_wired_socket_in_order(run_prompt) -> None:
    success, seen, errors = run_prompt(_flattened_subgraph_prompt(10))

    assert success, errors
    assert [name for name, _shape in seen[0]] == [f"image_{i}" for i in range(1, 11)]
    assert [shape[1] for _name, shape in seen[0]] == list(range(8, 18))


def test_fan_out_treats_a_bypassed_loader_as_an_empty_list(run_prompt) -> None:
    success, seen, errors = run_prompt(
        _flattened_subgraph_prompt(0, loader_bypassed=True)
    )

    assert success, errors
    assert seen == [[]]


def test_fan_out_rejects_items_that_would_land_on_unwired_outputs(run_prompt) -> None:
    success, seen, errors = run_prompt(_flattened_subgraph_prompt(12))

    assert not success
    assert seen == []
    assert "item_11, item_12 are not connected" in errors[0]


def test_fan_out_rejects_lists_longer_than_its_outputs(run_prompt) -> None:
    success, _seen, errors = run_prompt(_flattened_subgraph_prompt(17, wired=16))

    assert not success
    assert f"has {SLOT_COUNT} outputs; received 17 items" in errors[0]
