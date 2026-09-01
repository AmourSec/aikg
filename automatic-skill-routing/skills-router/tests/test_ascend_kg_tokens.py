from __future__ import annotations

import unittest

from runtime.contracts import (
    ActivationConsent,
    CandidateId,
    Candidates,
    InvalidResponseReason,
    LoadRequest,
    NetworkConsent,
    RemoteLoadInvalid,
    RemoteSkillContent,
    ResponseToken,
)
from tests.ascend_kg_support import (
    candidate_search_response,
    json_response,
    make_provider,
    search_request,
)


SKILL_BODY = b"---\nname: remote\ndescription: A remote test skill.\n---\n# Remote\nDo work.\n"


def _load_request(token: ResponseToken, candidate_id: str) -> LoadRequest:
    return LoadRequest(
        response_token=token,
        candidate_id=CandidateId(candidate_id),
        network_consent=NetworkConsent.GRANTED,
        activation_consent=ActivationConsent.GRANTED,
    )


class AscendKgResponseTokenTests(unittest.TestCase):
    def test_load_rejects_forged_token_before_transport(self) -> None:
        # Given
        provider, transport, _ = make_provider((candidate_search_response("remote-1"),))
        search_result = provider.search(search_request())
        assert isinstance(search_result, Candidates)
        forged = ResponseToken(candidate_ids=search_result.response_token.candidate_ids)

        # When
        result = provider.load_skill(_load_request(forged, "remote-1"))

        # Then
        self.assertEqual(
            result,
            RemoteLoadInvalid(InvalidResponseReason.CANDIDATE_MEMBERSHIP),
        )
        self.assertEqual(len(transport.requests), 1)

    def test_load_rejects_token_replaced_by_new_search(self) -> None:
        # Given
        provider, transport, _ = make_provider(
            (candidate_search_response("old"), candidate_search_response("current")),
        )
        old_result = provider.search(search_request())
        current_result = provider.search(search_request())
        assert isinstance(old_result, Candidates)
        assert isinstance(current_result, Candidates)

        # When
        result = provider.load_skill(_load_request(old_result.response_token, "old"))

        # Then
        self.assertEqual(
            result,
            RemoteLoadInvalid(InvalidResponseReason.CANDIDATE_MEMBERSHIP),
        )
        self.assertEqual(len(transport.requests), 2)

    def test_search_without_candidates_revokes_previous_token(self) -> None:
        # Given
        provider, transport, _ = make_provider(
            (candidate_search_response("old"), json_response({"results": []})),
        )
        old_result = provider.search(search_request())
        provider.search(search_request())
        assert isinstance(old_result, Candidates)

        # When
        result = provider.load_skill(_load_request(old_result.response_token, "old"))

        # Then
        self.assertEqual(
            result,
            RemoteLoadInvalid(InvalidResponseReason.CANDIDATE_MEMBERSHIP),
        )
        self.assertEqual(len(transport.requests), 2)

    def test_load_accepts_each_member_of_current_multi_id_token(self) -> None:
        # Given
        provider, transport, _ = make_provider(
            (
                candidate_search_response("first", "second"),
                json_response(SKILL_BODY),
                json_response(SKILL_BODY),
            ),
        )
        search_result = provider.search(search_request())
        assert isinstance(search_result, Candidates)

        # When
        results = tuple(
            provider.load_skill(_load_request(search_result.response_token, candidate_id))
            for candidate_id in ("first", "second")
        )

        # Then
        self.assertTrue(all(isinstance(result, RemoteSkillContent) for result in results))
        self.assertEqual(len(transport.requests), 3)


if __name__ == "__main__":
    unittest.main()
