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


LIST_FAN_OUT_SLOTS = 16


def _connected_output_slots(dynprompt, node_id: str) -> set[int] | None:
    if dynprompt is None or node_id is None:
        return None
    connected: set[int] = set()
    for consumer_id in dynprompt.all_node_ids():
        consumer = dynprompt.get_node(consumer_id)
        for value in consumer.get("inputs", {}).values():
            if (
                isinstance(value, list)
                and len(value) == 2
                and str(value[0]) == str(node_id)
                and isinstance(value[1], int)
            ):
                connected.add(value[1])
    return connected


class vloListFanOut(io.ComfyNode):
    """Spread a Comfy list across fixed sockets, padding the tail with None.

    Nodes with numbered optional inputs (an Autogrow group, or a subgraph that
    promotes one) treat a None value like an unconnected socket, so one fan-out
    replaces a per-model batch adapter. The shared MatchType template makes the
    connected list's type the type of every output.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        template = io.MatchType.Template("value")
        return io.Schema(
            node_id="vloListFanOut",
            search_aliases=["fan out", "list to outputs", "unbatch", "split list"],
            display_name="vlo List Fan Out",
            category="utils/logic",
            description=(
                "Sends item N of a list to output N. Outputs past the end of the "
                "list are None, which nodes with optional inputs read as "
                "unconnected. An absent list (for example a bypassed loader) "
                "makes every output None."
            ),
            is_input_list=True,
            inputs=[
                io.MatchType.Input(
                    "values",
                    template=template,
                    optional=True,
                    tooltip="Ordered list to spread across the outputs.",
                ),
            ],
            outputs=[
                io.MatchType.Output(
                    template=template,
                    id=f"item_{index}",
                    display_name=f"item_{index}",
                )
                for index in range(1, LIST_FAN_OUT_SLOTS + 1)
            ],
            hidden=[io.Hidden.unique_id, io.Hidden.dynprompt],
        )

    @classmethod
    def execute(cls, values=None) -> io.NodeOutput:
        items = [] if values is None else list(values)
        if len(items) > LIST_FAN_OUT_SLOTS:
            raise ValueError(
                f"vlo List Fan Out has {LIST_FAN_OUT_SLOTS} outputs; "
                f"received {len(items)} items"
            )
        # An item whose output nobody reads would vanish silently (for example
        # a 12-image list into a 10-input subgraph), so treat it as an error.
        connected = _connected_output_slots(
            cls.hidden.dynprompt, cls.hidden.unique_id
        )
        if connected is not None:
            dropped = [
                index + 1 for index in range(len(items)) if index not in connected
            ]
            if dropped:
                raise ValueError(
                    f"vlo List Fan Out received {len(items)} items but "
                    f"output(s) {', '.join(f'item_{i}' for i in dropped)} "
                    "are not connected"
                )
        padded = items + [None] * (LIST_FAN_OUT_SLOTS - len(items))
        return io.NodeOutput(*padded)


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
