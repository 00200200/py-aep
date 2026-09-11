"""Layer enumeration of Illustrator/PDF Optional Content Groups.

The `.ai` fixtures under `samples/assets/` are well-formed Illustrator
output, where `/OCGs` and the default configuration's `/Order` hold the same
groups. Real files are not always that tidy - Illustrator can leave the
groups of earlier revisions of the artwork behind in `/OCGs` - so the cases
here are hand-built PDF bodies that isolate one structure each.
"""

from __future__ import annotations

import pytest

from py_aep.resolvers.ai_layers import (
    AiLayer,
    UnsupportedAiLayersError,
    read_ai_layer_ocgs,
    read_ai_layers,
)


def _pdf(oc_properties: str) -> bytes:
    """A minimal PDF body whose catalog carries `oc_properties`."""
    return (
        "%PDF-1.5\n"
        "1 0 obj\n"
        "<</Type/Catalog/OCProperties" + oc_properties + ">>\n"
        "endobj\n"
        "10 0 obj\n<</Type/OCG/Name(Stale)>>\nendobj\n"
        "11 0 obj\n<</Type/OCG/Name(Hidden)>>\nendobj\n"
        "12 0 obj\n<</Type/OCG/Name(Visible)>>\nendobj\n"
        # The parser reads one token past an object, so the last group needs
        # something behind it, as it has in a real document.
        "13 0 obj\nnull\nendobj\n"
    ).encode("ascii")


class TestStaleGroups:
    """`/OCGs` can list groups the document no longer has."""

    def test_groups_absent_from_order_are_dropped(self) -> None:
        # A 2019-vintage Illustrator file shipped 65 groups for its 25
        # layers, the 40 stale ones first; every layer index After Effects
        # had authored against it only lined up with the /Order members.
        data = _pdf("<</OCGs[10 0 R 11 0 R 12 0 R]/D<</Order[12 0 R 11 0 R]>>>>")

        assert read_ai_layer_ocgs("probe.ai", data) == [
            AiLayer(11, "Hidden", True),
            AiLayer(12, "Visible", True),
        ]

    def test_order_decides_membership_not_order(self) -> None:
        """`/OCGs` stays the document order even where `/Order` reverses it."""
        data = _pdf("<</OCGs[12 0 R 11 0 R]/D<</Order[11 0 R 12 0 R]>>>>")

        assert read_ai_layers("probe.ai", data) == ["Visible", "Hidden"]

    def test_nested_order_entries_count_as_members(self) -> None:
        """Illustrator sublayers nest, with an optional label string first."""
        data = _pdf("<</OCGs[10 0 R 11 0 R 12 0 R]/D<</Order[(Set)[12 0 R] 11 0 R]>>>>")

        assert read_ai_layers("probe.ai", data) == ["Hidden", "Visible"]

    def test_a_file_without_order_keeps_every_group(self) -> None:
        data = _pdf("<</OCGs[10 0 R 11 0 R 12 0 R]/D<</BaseState/ON>>>>")

        assert read_ai_layers("probe.ai", data) == ["Stale", "Hidden", "Visible"]

    def test_no_group_survives_the_filter_raises(self) -> None:
        data = _pdf("<</OCGs[10 0 R]/D<</Order[99 0 R]>>>>")

        with pytest.raises(UnsupportedAiLayersError, match="no named layers"):
            read_ai_layers("probe.ai", data)


class TestVisibility:
    """`/D` `/OFF` lists the groups the default configuration hides."""

    def test_a_group_in_off_is_not_visible(self) -> None:
        data = _pdf("<</OCGs[11 0 R 12 0 R]/D<</Order[12 0 R 11 0 R]/OFF[11 0 R]>>>>")

        assert read_ai_layer_ocgs("probe.ai", data) == [
            AiLayer(11, "Hidden", False),
            AiLayer(12, "Visible", True),
        ]

    def test_every_group_is_visible_without_off(self) -> None:
        data = _pdf("<</OCGs[11 0 R 12 0 R]/D<</Order[12 0 R 11 0 R]>>>>")

        assert [layer.visible for layer in read_ai_layer_ocgs("probe.ai", data)] == [
            True,
            True,
        ]
