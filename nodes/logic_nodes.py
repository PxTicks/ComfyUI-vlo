"""Small graph-control nodes."""

from __future__ import annotations

from comfy_api.latest import io


class vloGateNone(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        template = io.MatchType.Template("value")
        return io.Schema(
            node_id="vloGateNone",
            search_aliases=["gate", "null gate", "disable pass-through", "none gate"],
            display_name="vlo Gate None",
            category="utils/logic",
            description=(
                "Passes any connected value through unchanged unless disabled is true, "
                "in which case the output is None."
            ),
            inputs=[
                io.MatchType.Input(
                    "value",
                    template=template,
                    tooltip="Any connected value to pass through or suppress.",
                ),
                io.Boolean.Input(
                    "disabled",
                    default=False,
                    tooltip="When true, suppresses the value and outputs None instead.",
                ),
            ],
            outputs=[
                io.MatchType.Output(
                    template=template,
                    display_name="value",
                )
            ],
        )

    @classmethod
    def execute(cls, value, disabled=False) -> io.NodeOutput:
        return io.NodeOutput(None if disabled else value)


class vloLogicNot(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="vloLogicNot",
            search_aliases=["not", "invert boolean", "logic not", "boolean not"],
            display_name="vlo Logic Not",
            category="utils/logic",
            description="Inverts an incoming boolean value.",
            inputs=[
                io.Boolean.Input(
                    "value",
                    force_input=True,
                    tooltip="The boolean value to invert.",
                ),
            ],
            outputs=[
                io.Boolean.Output(display_name="value")
            ],
        )

    @classmethod
    def execute(cls, value) -> io.NodeOutput:
        return io.NodeOutput(not value)
