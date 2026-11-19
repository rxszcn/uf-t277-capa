# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
round-trip tests for the result document / match structure contract,
built on fully synthetic fixtures.

unlike tests/test_result_document.py, nothing here reads the external
golden-sample directory (tests/data/rd): the rule sources, the extracted
feature sets, and the expected captures are all constructed inline.
this keeps the serialization round-trip and the match-splicing logic
runnable in snapshot environments where the golden samples are not
initialized.

coverage, each with a positive and a negative (tampered) form:
  1. result document JSON serialization/deserialization is byte-identical
     across a round trip.
  2. match structures rebuilt via ResultDocument.to_capa() are consistent
     with the match trees stored in the document.
  3. match entries splice referenced rule/namespace matches into the
     parent match tree, driven by match locations.
  4. capture groups (regex/substring matches) survive the round trip.
"""

import copy
import textwrap
import datetime
import collections

import pytest

import capa.rules
import capa.engine
import capa.features.common
import capa.features.address
import capa.features.freeze as frz
import capa.features.freeze.features as frzf
import capa.render.result_document as rdoc

# ---------------------------------------------------------------------------
# synthetic fixture constants
# ---------------------------------------------------------------------------

CHILD_RULE_NAME = "synthetic child capability"
CHILD_NAMESPACE = "synthetic/child"
PARENT_BY_NAMESPACE_NAME = "synthetic parent via namespace"
PARENT_BY_RULE_NAME = "synthetic parent via rule name"

PARENT_KINDS = ["namespace", "rule"]
PARENT_RULE_NAMES = {
    "namespace": PARENT_BY_NAMESPACE_NAME,
    "rule": PARENT_BY_RULE_NAME,
}
PARENT_MATCH_TARGETS = {
    "namespace": CHILD_NAMESPACE,
    "rule": CHILD_RULE_NAME,
}

# addresses are chosen to be unique across the whole document,
# so that a JSON-level tamper of one value hits exactly one logical field.
ADDR_BASE = 0x400000
ADDR_STRING_1 = 0x401000
ADDR_STRING_2 = 0x401004
ADDR_FUNCTION = 0x402000
ADDR_BASIC_BLOCK = 0x402100
ADDR_LIBRARY = 0x403000
ADDR_MISALIGNED = 0x404000

STRING_1 = "prefix synthetic_payload_marker suffix"
STRING_2 = "synthetic_implant_marker"
TAMPERED_STRING = "synthetic_rootkit_marker"

VERSION = "9.9.9-synthetic"

NO_ADDRESS = capa.features.address.NO_ADDRESS
AVA = capa.features.address.AbsoluteVirtualAddress

EXPECTED_CAPTURES = {
    STRING_1: (frz.Address.from_capa(AVA(ADDR_STRING_1)),),
    STRING_2: (frz.Address.from_capa(AVA(ADDR_STRING_2)),),
}

EXPECTED_CAPA_MATCHES = {
    STRING_1: {AVA(ADDR_STRING_1)},
    STRING_2: {AVA(ADDR_STRING_2)},
}


# ---------------------------------------------------------------------------
# synthetic rule sources
# ---------------------------------------------------------------------------


def make_child_rule_source() -> str:
    return textwrap.dedent(
        f"""\
        rule:
          meta:
            name: {CHILD_RULE_NAME}
            namespace: {CHILD_NAMESPACE}
            authors:
              - synthetic-tester@example.com
            scopes:
              static: file
              dynamic: file
            att&ck:
              - Execution::Command and Scripting Interpreter::Python [T1059.006]
            mbc:
              - Collection::Input Capture::Mouse Events [E1056.m01]
            references:
              - https://example.com/synthetic/child
            examples:
              - 0123456789abcdef
            description: synthetic child rule for round-trip fixtures
          features:
            - and:
              - string: /synthetic_(payload|implant)_marker/
              - substring: marker
        """
    )


def make_parent_rule_source(parent_kind: str) -> str:
    return textwrap.dedent(
        f"""\
        rule:
          meta:
            name: {PARENT_RULE_NAMES[parent_kind]}
            authors:
              - synthetic-tester@example.com
            scopes:
              static: file
              dynamic: file
            description: synthetic parent rule for match/namespace splicing fixtures
          features:
            - and:
              - match: {PARENT_MATCH_TARGETS[parent_kind]}
        """
    )


# ---------------------------------------------------------------------------
# synthetic fixture builders
# ---------------------------------------------------------------------------


def make_ruleset(parent_kind: str) -> capa.rules.RuleSet:
    rules = [
        capa.rules.Rule.from_yaml(make_child_rule_source()),
        capa.rules.Rule.from_yaml(make_parent_rule_source(parent_kind)),
    ]
    return capa.rules.RuleSet(rules)


def make_features() -> capa.engine.FeatureSet:
    # the synthetic "extractor" output: two strings carrying the markers
    # that the child rule's regex and substring features fire on.
    return collections.defaultdict(
        set,
        {
            capa.features.common.String(STRING_1): {AVA(ADDR_STRING_1)},
            capa.features.common.String(STRING_2): {AVA(ADDR_STRING_2)},
        },
    )


def make_static_metadata() -> rdoc.Metadata:
    return rdoc.StaticMetadata(
        timestamp=datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc),
        version=VERSION,
        argv=("capa", "--json", "synthetic.bin"),
        sample=rdoc.Sample(
            md5="0" * 32,
            sha1="0" * 40,
            sha256="0" * 64,
            path="synthetic.bin",
        ),
        analysis=rdoc.StaticAnalysis(
            format="pe",
            arch="i386",
            os="windows",
            extractor="SyntheticExtractor",
            rules=("synthetic-rules/",),
            base_address=frz.Address.from_capa(AVA(ADDR_BASE)),
            layout=rdoc.StaticLayout(
                functions=(
                    rdoc.FunctionLayout(
                        address=frz.Address.from_capa(AVA(ADDR_FUNCTION)),
                        matched_basic_blocks=(
                            rdoc.BasicBlockLayout(address=frz.Address.from_capa(AVA(ADDR_BASIC_BLOCK))),
                        ),
                    ),
                )
            ),
            feature_counts=rdoc.StaticFeatureCounts(
                file=2,
                functions=(
                    rdoc.FunctionFeatureCount(
                        address=frz.Address.from_capa(AVA(ADDR_FUNCTION)),
                        count=2,
                    ),
                ),
            ),
            library_functions=(
                rdoc.LibraryFunction(
                    address=frz.Address.from_capa(AVA(ADDR_LIBRARY)),
                    name="synthetic_library_function",
                ),
            ),
        ),
    )


def canonicalize(doc: rdoc.ResultDocument) -> rdoc.ResultDocument:
    """
    round a freshly built document through the storage contract once,
    returning the canonical form that a reader of the store observes.

    serialization normalizes some types (e.g. the Metadata subclasses
    deserialize as the Metadata base class), so all assertions below
    are made against the canonical, read-back form of the document.
    """
    return rdoc.ResultDocument.model_validate_json(doc.model_dump_json(exclude_none=True))


def make_result_document(parent_kind: str) -> rdoc.ResultDocument:
    ruleset = make_ruleset(parent_kind)
    _, capabilities = ruleset.match(capa.rules.Scope.FILE, make_features(), NO_ADDRESS)

    # fixture sanity: both the child and the parent rule must fire,
    # otherwise the downstream assertions would be vacuous.
    assert CHILD_RULE_NAME in capabilities
    assert PARENT_RULE_NAMES[parent_kind] in capabilities

    doc = rdoc.ResultDocument.from_capa(make_static_metadata(), ruleset, capabilities)
    return canonicalize(doc)


def make_misaligned_result_document() -> rdoc.ResultDocument:
    """
    bad-form fixture: the parent rule's `match` feature is recorded at a
    location (ADDR_MISALIGNED) that does not intersect the locations at
    which the child rule actually matched (NO_ADDRESS).

    this models the "evidence field misalignment" failure mode: the child
    rule's match evidence exists in the document, but the splice into the
    parent's match tree is location-driven, so the evidence is silently
    dropped from the parent's tree.
    """
    parent_kind = "namespace"
    ruleset = make_ruleset(parent_kind)
    _, capabilities = ruleset.match(capa.rules.Scope.FILE, make_features(), NO_ADDRESS)

    parent_rule = ruleset.rules[PARENT_RULE_NAMES[parent_kind]]
    misaligned_features = collections.defaultdict(
        set,
        {capa.features.common.MatchedRule(CHILD_NAMESPACE): {AVA(ADDR_MISALIGNED)}},
    )
    parent_result = parent_rule.evaluate(misaligned_features, short_circuit=False)
    assert bool(parent_result) is True

    capabilities = collections.defaultdict(list, {name: list(matches) for name, matches in capabilities.items()})
    capabilities[PARENT_RULE_NAMES[parent_kind]] = [(NO_ADDRESS, parent_result)]

    doc = rdoc.ResultDocument.from_capa(make_static_metadata(), ruleset, capabilities)
    return canonicalize(doc)


# ---------------------------------------------------------------------------
# tampering helpers (bad-form variants)
# ---------------------------------------------------------------------------


def tamper_version(payload: str) -> str:
    # metadata field corruption: the producer version no longer round-trips.
    return payload.replace(f'"version":"{VERSION}"', '"version":"0.0.0-corrupt"', 1)


def tamper_location(payload: str) -> str:
    # evidence field misalignment: one match location is shifted by one byte.
    old = f'"value":{ADDR_STRING_1}'
    new = f'"value":{ADDR_STRING_1 + 1}'
    assert old in payload
    return payload.replace(old, new, 1)


def tamper_capture(payload: str) -> str:
    # capture group corruption: a captured string is rewritten.
    assert STRING_2 in payload
    return payload.replace(STRING_2, TAMPERED_STRING, 1)


def tamper_success(payload: str) -> str:
    # verdict flip: the first (root) match node is recorded as a failure.
    assert '"success":true' in payload
    return payload.replace('"success":true', '"success":false', 1)


# ---------------------------------------------------------------------------
# structural comparison helpers
# ---------------------------------------------------------------------------


def iter_match_tree(match: rdoc.Match):
    yield match
    for child in match.children:
        yield from iter_match_tree(child)


def iter_result_tree(result: capa.features.common.Result):
    yield result
    for child in result.children:
        yield from iter_result_tree(child)


def get_result(capabilities, rule_name: str, addr: capa.features.address.Address) -> capa.features.common.Result:
    for result_addr, result in capabilities.matches[rule_name]:
        if result_addr == addr:
            return result
    raise AssertionError(f"no result for rule {rule_name!r} at address {addr}")


def assert_match_reconstructed(match: rdoc.Match, result: capa.features.common.Result) -> None:
    """
    assert that the engine Result rebuilt via ResultDocument.to_capa()
    is structurally consistent with the rdoc.Match it was rebuilt from:
    same verdict, same locations, same node kind, same captures,
    and pairwise-consistent children.
    """
    assert result.success == match.success
    assert {frz.Address.from_capa(loc) for loc in result.locations} == set(match.locations)

    node = match.node
    if isinstance(node, rdoc.StatementNode):
        statement = node.statement
        if isinstance(statement, rdoc.CompoundStatement):
            expected_types = {
                rdoc.CompoundStatementType.AND: capa.engine.And,
                rdoc.CompoundStatementType.OR: capa.engine.Or,
                rdoc.CompoundStatementType.NOT: capa.engine.Not,
                rdoc.CompoundStatementType.OPTIONAL: capa.engine.Some,
            }
            assert isinstance(result.statement, expected_types[statement.type])
        elif isinstance(statement, rdoc.SomeStatement):
            assert isinstance(result.statement, capa.engine.Some)
            assert result.statement.count == statement.count
        elif isinstance(statement, rdoc.RangeStatement):
            assert isinstance(result.statement, capa.engine.Range)
            assert result.statement.min == statement.min
            assert result.statement.max == statement.max
        elif isinstance(statement, rdoc.SubscopeStatement):
            assert isinstance(result.statement, capa.engine.Subscope)
            assert result.statement.scope == statement.scope
        else:
            raise AssertionError(f"unexpected statement model: {statement!r}")
        assert result.statement.description == statement.description
    else:
        assert isinstance(node, rdoc.FeatureNode)
        expected_feature = node.feature.to_capa()
        # Feature.__eq__ compares the feature name and value.
        assert result.statement == expected_feature
        if isinstance(
            result.statement,
            (capa.features.common._MatchedSubstring, capa.features.common._MatchedRegex),
        ):
            expected_matches = {
                capture: {loc.to_capa() for loc in locations} for capture, locations in match.captures.items()
            }
            assert result.statement.matches == expected_matches

    assert len(result.children) == len(match.children)
    for child_match, child_result in zip(match.children, result.children):
        assert_match_reconstructed(child_match, child_result)


def find_match_feature_matches(match: rdoc.Match) -> list[rdoc.Match]:
    return [
        node_match
        for node_match in iter_match_tree(match)
        if isinstance(node_match.node, rdoc.FeatureNode) and isinstance(node_match.node.feature, frzf.MatchFeature)
    ]


# ---------------------------------------------------------------------------
# coverage 1: result document JSON serialization round trip (byte-identical)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("parent_kind", PARENT_KINDS)
def test_json_roundtrip_byte_identical(parent_kind):
    doc = make_result_document(parent_kind)

    payload = doc.model_dump_json(exclude_none=True)
    doc2 = rdoc.ResultDocument.model_validate_json(payload)
    payload2 = doc2.model_dump_json(exclude_none=True)

    # object-level equality via pydantic model equality,
    assert doc2 == doc
    # and byte-level equality of the serialized JSON documents.
    assert payload2 == payload
    assert payload2.encode("utf-8") == payload.encode("utf-8")


@pytest.mark.parametrize("parent_kind", PARENT_KINDS)
@pytest.mark.parametrize(
    "tamper",
    [tamper_version, tamper_location, tamper_capture],
    ids=lambda t: t.__name__,
)
def test_json_roundtrip_detects_tampered_document(parent_kind, tamper):
    doc = make_result_document(parent_kind)
    payload = doc.model_dump_json(exclude_none=True)

    bad_payload = tamper(payload)
    assert bad_payload != payload

    bad_doc = rdoc.ResultDocument.model_validate_json(bad_payload)

    # the tampered document must not compare equal to the original,
    # and its serialization must differ byte-for-byte:
    # a corrupted field can never alias the original document.
    assert bad_doc != doc
    assert bad_doc.model_dump_json(exclude_none=True) != payload


# ---------------------------------------------------------------------------
# coverage 2: match structure reconstruction (ResultDocument.to_capa)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("parent_kind", PARENT_KINDS)
def test_match_structure_reconstruction(parent_kind):
    doc = make_result_document(parent_kind)

    meta, capabilities = doc.to_capa()
    assert meta == doc.meta

    for rule_name, rule_matches in doc.rules.items():
        for addr, match in rule_matches.matches:
            result = get_result(capabilities, rule_name, addr.to_capa())
            assert_match_reconstructed(match, result)


@pytest.mark.parametrize("parent_kind", PARENT_KINDS)
@pytest.mark.parametrize(
    "tamper",
    [tamper_success, tamper_location],
    ids=lambda t: t.__name__,
)
def test_match_structure_reconstruction_detects_tampered_document(parent_kind, tamper):
    doc = make_result_document(parent_kind)
    bad_doc = rdoc.ResultDocument.model_validate_json(tamper(doc.model_dump_json(exclude_none=True)))

    _, bad_capabilities = bad_doc.to_capa()

    # the structures rebuilt from the tampered document must NOT be
    # consistent with the original document's match trees.
    with pytest.raises(AssertionError):
        for rule_name, rule_matches in doc.rules.items():
            for addr, match in rule_matches.matches:
                result = get_result(bad_capabilities, rule_name, addr.to_capa())
                assert_match_reconstructed(match, result)


# ---------------------------------------------------------------------------
# coverage 3: match entries and namespace/rule splicing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parent_kind,expected_target",
    [
        pytest.param("namespace", CHILD_NAMESPACE, id="namespace"),
        pytest.param("rule", CHILD_RULE_NAME, id="rule-name"),
    ],
)
def test_match_entry_splices_referenced_matches(parent_kind, expected_target):
    doc = make_result_document(parent_kind)

    parent = doc.rules[PARENT_RULE_NAMES[parent_kind]]
    assert len(parent.matches) == 1
    addr, match = parent.matches[0]
    # file-scope rules match at NO_ADDRESS.
    assert addr == frz.Address.from_capa(NO_ADDRESS)

    match_features = find_match_feature_matches(match)
    assert len(match_features) == 1
    match_feature = match_features[0]

    # the match entry records the referenced rule name / namespace...
    assert isinstance(match_feature.node, rdoc.FeatureNode)
    assert isinstance(match_feature.node.feature, frzf.MatchFeature)
    assert match_feature.node.feature.match == expected_target
    # ...at the location where the referenced rule matched...
    assert match_feature.locations == (frz.Address.from_capa(NO_ADDRESS),)

    # ...and the referenced rule's own match tree is spliced in, verbatim.
    child_match = doc.rules[CHILD_RULE_NAME].matches[0][1]
    assert match_feature.children == (child_match,)


def test_match_entry_splice_drops_misaligned_locations():
    # bad form: the parent's match entry is recorded at a location that
    # does not intersect the child rule's match locations.
    doc = make_misaligned_result_document()

    # the child rule's own evidence is present in the document...
    child = doc.rules[CHILD_RULE_NAME]
    assert len(child.matches) == 1

    parent = doc.rules[PARENT_BY_NAMESPACE_NAME]
    assert len(parent.matches) == 1
    _, match = parent.matches[0]

    match_features = find_match_feature_matches(match)
    assert len(match_features) == 1
    match_feature = match_features[0]

    # ...but the parent's match entry points at the misaligned location...
    assert match_feature.locations == (frz.Address.from_capa(AVA(ADDR_MISALIGNED)),)
    # ...so the splice finds no matching child evidence and the parent's
    # tree silently loses the referenced match subtree.
    assert match_feature.children == ()


# ---------------------------------------------------------------------------
# coverage 4: capture group round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("parent_kind", PARENT_KINDS)
def test_capture_groups_roundtrip(parent_kind):
    doc = make_result_document(parent_kind)

    child_match = doc.rules[CHILD_RULE_NAME].matches[0][1]
    capture_nodes = [node_match for node_match in iter_match_tree(child_match) if node_match.captures]

    # both the regex and the substring feature must record captures.
    feature_types = {type(node_match.node.feature) for node_match in capture_nodes}
    assert frzf.RegexFeature in feature_types
    assert frzf.SubstringFeature in feature_types
    for node_match in capture_nodes:
        assert node_match.captures == EXPECTED_CAPTURES

    # captures survive the JSON round trip unchanged.
    doc2 = rdoc.ResultDocument.model_validate_json(doc.model_dump_json(exclude_none=True))
    child_match2 = doc2.rules[CHILD_RULE_NAME].matches[0][1]
    captures2 = [node_match.captures for node_match in iter_match_tree(child_match2) if node_match.captures]
    assert captures2 == [node_match.captures for node_match in capture_nodes]

    # captures are rebuilt into _MatchedRegex/_MatchedSubstring statements.
    _, capabilities = doc.to_capa()
    result = get_result(capabilities, CHILD_RULE_NAME, NO_ADDRESS)
    matched_statements = [
        r.statement
        for r in iter_result_tree(result)
        if isinstance(r.statement, (capa.features.common._MatchedSubstring, capa.features.common._MatchedRegex))
        and r.statement.matches
    ]
    assert len(matched_statements) == 2
    for statement in matched_statements:
        assert statement.matches == EXPECTED_CAPA_MATCHES


@pytest.mark.parametrize("parent_kind", PARENT_KINDS)
def test_capture_groups_roundtrip_detects_tampered_capture(parent_kind):
    doc = make_result_document(parent_kind)
    bad_doc = rdoc.ResultDocument.model_validate_json(tamper_capture(doc.model_dump_json(exclude_none=True)))

    # the tampered capture must be visible in the document...
    assert bad_doc != doc
    child_match = bad_doc.rules[CHILD_RULE_NAME].matches[0][1]
    bad_captures = [node_match.captures for node_match in iter_match_tree(child_match) if node_match.captures]
    assert bad_captures
    assert any(captures != EXPECTED_CAPTURES for captures in bad_captures)
    assert any(TAMPERED_STRING in captures for captures in bad_captures)

    # ...and in the structures rebuilt from it.
    _, bad_capabilities = bad_doc.to_capa()
    result = get_result(bad_capabilities, CHILD_RULE_NAME, NO_ADDRESS)
    bad_matches = [
        r.statement.matches
        for r in iter_result_tree(result)
        if isinstance(r.statement, (capa.features.common._MatchedSubstring, capa.features.common._MatchedRegex))
        and r.statement.matches
    ]
    assert bad_matches
    assert any(matches != EXPECTED_CAPA_MATCHES for matches in bad_matches)
    assert any(TAMPERED_STRING in matches for matches in bad_matches)
