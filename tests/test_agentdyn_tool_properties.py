"""Verify the only new decision inputs are tool-effect metadata."""
from dataclasses import asdict
from types import SimpleNamespace
import unittest

from obligate.eval.agentdyn.tool_properties import ACTIVE_CALL, AgentDynToolTaxonomy
from obligate.eval.agentdojo.evidence.taxonomy import default_specs


class ToolPropertyTests(unittest.TestCase):
    def setUp(self):
        self.taxonomy = AgentDynToolTaxonomy()

    def test_effects_not_name_prefixes(self):
        self.assertTrue(self.taxonomy.classify("checkout_selected_cart").side_effect)
        self.assertEqual(self.taxonomy.classify("git_push").group, "external_send")
        self.assertEqual(self.taxonomy.classify("git_transfer_repo_ownership").group, "membership_mutation")
        self.assertEqual(self.taxonomy.classify("verify_transaction").group, "financial_commit")
        self.assertTrue(self.taxonomy.classify("get_shopping_linked_unread_emails").state_mutation)

    def test_core_specs_preserved(self):
        for name in ("send_money", "send_email", "read_file", "create_calendar_event"):
            self.assertEqual(asdict(self.taxonomy.classify(name)), asdict(default_specs()[name]))

    def test_received_email_docstring_does_not_override_actual_effect(self):
        self.assertFalse(self.taxonomy.classify("get_shopping_linked_received_emails").side_effect)
        self.assertTrue(self.taxonomy.classify("get_shopping_linked_unread_emails").side_effect)
        self.assertTrue(self.taxonomy.classify("get_unread_emails").state_mutation)

    def test_runtime_web_effect_does_not_read_content_or_static_arguments(self):
        # Both fields deliberately fail if the adapter attempts to inspect them.
        class Handler:
            handler = "send_money"
            @property
            def static_parameters(self):
                raise AssertionError("handler values must not enter tool classification")

        class Web:
            web_url = "example.test/action"
            field_to_parameters = {}
            web_function = [Handler()]
            @property
            def web_content(self):
                raise AssertionError("page content must not enter tool classification")

        env = SimpleNamespace(web_database=SimpleNamespace(web_list=[Web()]))
        token = ACTIVE_CALL.set((env, {"url": "https://example.test/action"}))
        try:
            effect = self.taxonomy.classify("browse_webpage")
            self.assertEqual(effect.group, "financial_commit")
            self.assertTrue(effect.side_effect)
            self.assertEqual(effect.name, "browse_webpage")
        finally:
            ACTIVE_CALL.reset(token)
        # Call-local metadata never contaminates another case.
        self.assertEqual(self.taxonomy.classify("browse_webpage").group, "external_send")

    def test_form_page_visit_is_read_until_submitted(self):
        web = SimpleNamespace(web_url="example.test/form", field_to_parameters={"field": "password"},
                              web_function=[SimpleNamespace(handler="update_github_account_password")])
        env = SimpleNamespace(web_database=SimpleNamespace(web_list=[web]))
        token = ACTIVE_CALL.set((env, {"url": "example.test/form"}))
        try:
            self.assertFalse(self.taxonomy.classify("browse_webpage").side_effect)
            self.assertEqual(self.taxonomy.classify("input_to_webpage").group, "credential_change")
        finally:
            ACTIVE_CALL.reset(token)


if __name__ == "__main__":
    unittest.main()
