"""Tool-effect metadata, derived from upstream implementations, never attack goals.

Web dispatch metadata is inspected only to determine the tool's effect class.
Neither page text, handler arguments nor task/checker data enter this mapping.
"""
from contextvars import ContextVar
from dataclasses import replace

from obligate.eval.agentdojo.evidence.taxonomy import AgentDojoToolTaxonomy, default_specs, spec

ACTIVE_CALL = ContextVar("agentdyn_active_call", default=None)


def agentdyn_specs():
    tools = default_specs()

    def add(names, action, group, risk="low", **flags):
        for name in names.split():
            tools[name] = spec(name, action, group, risk, **flags)

    add("get_shopping_account_user_information get_logged_in_github_user_information git_get_linked_ssh_keys",
        "read_private_profile", "private_read", "medium", read_private=True)
    add("view_order_history get_invoice view_cart cart_total view_available_discounts",
        "read_private_shopping_data", "private_read", read_private=True)
    add("search_product get_github_account_user_information get_github_repository_information",
        "read_untrusted_content", "untrusted_read", read_untrusted=True)
    add("get_shopping_linked_unread_emails get_github_linked_unread_emails get_unread_emails",
        "read_private_email", "private_read", read_private=True, read_untrusted=True,
        side_effect=True, state_mutation=True)
    add("get_shopping_linked_received_emails get_received_emails get_sent_emails search_emails",
        "read_private_email", "private_read", read_private=True, read_untrusted=True)
    add("list_directory", "read_workspace_metadata", "read_only")
    add("login_shopping_account login_github_account verify_shopping_account verify_github_account",
        "authenticate_account", "workspace_mutation", "medium", side_effect=True,
        state_mutation=True, sensitive_args=["username", "password", "otp"])
    add("update_shopping_account_password update_github_account_password git_add_ssh_key git_delete_ssh_key",
        "credential_change", "credential_change", "critical", side_effect=True,
        state_mutation=True, sensitive_args=["password", "ssh_key"])
    add("update_shopping_account_user_info", "update_user_profile", "workspace_mutation", "high",
        side_effect=True, state_mutation=True, sensitive_args=["email", "address", "bank_account_number"])
    add("refund_order checkout_selected_cart verify_transaction", "financial_transaction", "financial_commit",
        "critical", side_effect=True, state_mutation=True,
        sensitive_args=["order_id", "product_ids", "discount_code", "otp"])
    add("cart_add_product cart_remove_product clear_cart", "update_shopping_cart", "workspace_mutation",
        "medium", side_effect=True, state_mutation=True, sensitive_args=["product_id", "quantity"])
    add("download_file_through_url download_file_through_id git_clone git_pull",
        "write_workspace_file", "workspace_mutation", "medium", side_effect=True,
        state_mutation=True, read_untrusted=True, sensitive_args=["url", "file_id", "save_dir", "local_path"])
    add("create_dir copy_node move_node", "write_workspace_file", "workspace_mutation", "medium",
        side_effect=True, state_mutation=True, sensitive_args=["path", "source_path", "destination_path"])
    add("delete_node git_delete_repo", "delete_workspace_resource", "destructive_update", "high",
        side_effect=True, state_mutation=True, sensitive_args=["path", "repo_name"])
    add("git_create_repo git_star git_unstar", "update_repository", "workspace_mutation", "medium",
        side_effect=True, state_mutation=True, sensitive_args=["repo_name"])
    add("git_invite_collaborators git_transfer_repo_ownership", "modify_membership", "membership_mutation", "high",
        side_effect=True, state_mutation=True, external_sink=True,
        sensitive_args=["repo_name", "collaborator_email", "new_owner_email", "new_owner_username"])
    add("git_push git_issue upload_file_to_web", "send_external_content", "external_send", "high",
        side_effect=True, external_sink=True, sensitive_args=["local_path", "comment_body", "repo_name"])
    add("input_to_webpage", "submit_web_form", "external_send", "high",
        side_effect=True, external_sink=True, state_mutation=True,
        sensitive_args=["url", "inputs", "input_ids", "input_values"])
    # Fallback when no dispatch metadata is bound. A generic click can mutate state.
    add("browse_webpage", "web_interaction", "external_send", "high",
        side_effect=True, external_sink=True, read_untrusted=True, sensitive_args=["url"])
    return tools


class AgentDynToolTaxonomy(AgentDojoToolTaxonomy):
    def __init__(self):
        super().__init__(agentdyn_specs())

    def classify(self, tool_name, *, suite=None):
        base = super().classify(tool_name, suite=suite)
        call = ACTIVE_CALL.get()
        if tool_name not in {"browse_webpage", "input_to_webpage"} or call is None:
            return base
        env, kwargs = call
        from agentdojo.default_suites.v1.tools.dynamic_web_client import fetch_web_instance, standardize_url
        web = fetch_web_instance(env.web_database, standardize_url(str(kwargs.get("url", ""))))
        handlers = web.web_function
        executes = bool(handlers) and (tool_name == "input_to_webpage" or not web.field_to_parameters)
        if not executes:
            return spec(tool_name, "network_fetch_untrusted_web", "untrusted_read", "medium",
                        read_untrusted=True, external_sink=True, sensitive_args=["url"])
        # Upstream returns after its first handler. Only its effect flags are projected.
        effect = self.specs.get(handlers[0].handler)
        if effect is None:
            return base
        return replace(effect, name=tool_name, read_untrusted=True,
                       sensitive_args=list(base.sensitive_args), decision_hints=[])
