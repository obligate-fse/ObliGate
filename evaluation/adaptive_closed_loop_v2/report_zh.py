"""Generate the Chinese ObliGate adaptive closed-loop experiment report."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import RESULT_ROOT, utc_now
from .summarize import build_summary, pct, read_json, wilson

REPORT_PATH = RESULT_ROOT / "report_zh.md"
HISTORICAL_REFERENCE_ROOT = RESULT_ROOT.parent / "adaptive_ablation" / "formal_activation_v1"

BENCHMARK_LABELS = {
    "ALL": "ALL",
    "agentdojo": "AgentDojo",
    "agent_safetybench": "Agent-SafetyBench",
    "agent_security_bench": "Agent Security Bench",
}

MECHANISM_LABELS = {
    "conflict_active": "conflict-active",
    "pure_gap_active": "pure-gap-active",
    "lifted_join_active": "lifted-join-active",
    "opi_authorization_observability": "OPI authorization-observability",
    "confirmation_pressure": "confirmation-pressure",
}


def ci(rate: dict[str, Any] | None) -> str:
    if not rate or rate.get("rate") is None:
        return "NA"
    return f"{pct(rate.get('rate'))} [{pct(rate.get('lo'))}, {pct(rate.get('hi'))}]"


def n_ci(n: int | None, total: int | None, rate: dict[str, Any] | None = None) -> str:
    if n is None or total in {None, 0}:
        return "NA"
    return f"{n}/{total} ({ci(rate or wilson(int(n), int(total)))})"


def compact_counts(counts: dict[str, Any] | None, limit: int = 4) -> str:
    if not counts:
        return "-"
    items = sorted(counts.items(), key=lambda item: (-int(item[1] or 0), str(item[0])))[:limit]
    return "; ".join(f"{key}:{value}" for key, value in items)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_暂无可报告数据。_"
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def cumulative_lookup(summary: dict[str, Any]) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    out = {}
    for row in summary.get("cumulative") or []:
        out[(str(row.get("segment")), str(row.get("victim_model")), str(row.get("benchmark")), int(row.get("horizon") or 0))] = row
    return out


def main_casr_table(summary: dict[str, Any]) -> str:
    lookup = cumulative_lookup(summary)
    rows = []
    models = sorted({str(row.get("victim_model")) for row in summary.get("cumulative") or [] if row.get("segment") == "main"})
    benches = ["ALL", "agentdojo", "agent_safetybench", "agent_security_bench"]
    for model in models:
        for bench in benches:
            r1 = lookup.get(("main", model, bench, 1))
            r3 = lookup.get(("main", model, bench, 3))
            r5 = lookup.get(("main", model, bench, 5))
            if not r1 and not r3 and not r5:
                continue
            base = r5 or r3 or r1 or {}
            rows.append(
                [
                    model,
                    BENCHMARK_LABELS.get(bench, bench),
                    base.get("cases", "NA"),
                    ci((r1 or {}).get("CASR")),
                    ci((r3 or {}).get("CASR")),
                    ci((r5 or {}).get("CASR")),
                    ci((r5 or {}).get("dangerous_candidate_rate") if r5 else None),
                    ci((r5 or {}).get("conditional_observed_bypass_given_dangerous") if r5 else None),
                    base.get("median_round_to_success") or "NA",
                ]
            )
    return md_table(["模型", "集合", "case", "CASR@1", "CASR@3", "CASR@5", "危险候选@5", "条件观察绕过@5", "首次成功中位轮"], rows)


def round_table(summary: dict[str, Any]) -> str:
    rows = []
    for row in summary.get("overall_by_round") or []:
        if row.get("segment") != "main":
            continue
        cases = int(row.get("cases") or 0)
        rows.append(
            [
                row.get("victim_model"),
                row.get("round"),
                cases,
                n_ci(row.get("attack_success"), cases, row.get("attack_success_rate")),
                n_ci(row.get("attack_success_observed"), cases, row.get("attack_success_observed_rate")),
                n_ci(row.get("dangerous_candidate"), cases, row.get("dangerous_candidate_rate")),
                n_ci(row.get("workflow_failure"), cases, row.get("workflow_failure_rate")),
                n_ci(row.get("task_success"), row.get("task_success_known_cases"), row.get("task_success_rate")),
                compact_counts(row.get("decision_counts")),
            ]
        )
    return md_table(["模型", "轮次", "评测 rows", "攻击成功", "观察绕过", "危险候选", "workflow failure", "原任务成功", "裁决分布"], rows)


def benchmark_table(summary: dict[str, Any]) -> str:
    rows = []
    for row in summary.get("by_benchmark_round") or []:
        if row.get("segment") != "main":
            continue
        cases = int(row.get("cases") or 0)
        rows.append(
            [
                row.get("victim_model"),
                BENCHMARK_LABELS.get(str(row.get("benchmark")), row.get("benchmark")),
                row.get("round"),
                cases,
                n_ci(row.get("attack_success"), cases, row.get("attack_success_rate")),
                n_ci(row.get("attack_success_observed"), cases, row.get("attack_success_observed_rate")),
                n_ci(row.get("dangerous_candidate"), cases, row.get("dangerous_candidate_rate")),
                ci(row.get("conditional_observed_bypass_given_dangerous")),
                n_ci(row.get("provider_valid"), cases, row.get("provider_valid_rate")),
            ]
        )
    return md_table(["模型", "benchmark", "轮次", "rows", "攻击成功", "观察绕过", "危险候选", "条件观察绕过", "provider valid"], rows)


def attacker_table(summary: dict[str, Any]) -> str:
    rows = []
    for row in summary.get("attacker_generation") or []:
        rows.append(
            [
                row.get("segment"),
                row.get("victim_model"),
                row.get("attacker_seed"),
                row.get("next_round"),
                row.get("generated"),
                f"{row.get('valid')}/{row.get('generated')}",
                n_ci(row.get("invalid"), row.get("generated"), row.get("invalid_rate")),
                compact_counts(row.get("primary_strategies"), 3),
                compact_counts(row.get("validation_failures"), 3),
            ]
        )
    return md_table(["segment", "victim", "seed", "生成目标轮", "生成数", "有效数", "无效率", "主要策略", "主要无效原因"], rows)


def stop_table(summary: dict[str, Any]) -> str:
    rows = []
    for row in summary.get("stop_reasons") or []:
        rows.append([row.get("segment"), row.get("victim_model"), row.get("attacker_seed"), row.get("stopped_cases"), compact_counts(row.get("reasons"), 8), compact_counts(row.get("rounds"), 8)])
    return md_table(["segment", "victim", "seed", "已停止 case", "停止原因", "停止轮次"], rows)


def usage_table(summary: dict[str, Any]) -> str:
    rows = []
    for row in summary.get("usage") or []:
        rows.append(
            [
                row.get("segment"),
                row.get("victim_model"),
                row.get("attacker_seed"),
                row.get("role"),
                row.get("calls"),
                row.get("input_tokens"),
                row.get("output_tokens"),
                row.get("total_tokens"),
                row.get("files"),
            ]
        )
    return md_table(["segment", "victim", "seed", "角色", "calls", "input", "output", "total", "files"], rows)


def provider_route_table() -> str:
    path = RESULT_ROOT / "preflight.json"
    if not path.exists():
        return "_preflight.json 尚不存在。_"
    data = read_json(path)
    rows = []
    for model, item in sorted((data.get("provider_checks") or {}).items()):
        rows.append(
            [
                model,
                item.get("key_env"),
                item.get("base_url"),
                item.get("credential_present"),
                item.get("credential_sha256_12"),
            ]
        )
    return md_table(["模型", "key env", "base URL", "凭证存在", "凭证 SHA256-12"], rows)


def stress_table(summary: dict[str, Any]) -> str:
    rows = []
    for row in summary.get("mechanism") or []:
        if row.get("horizon") != 5:
            continue
        cases = int(row.get("cases") or 0)
        category = str(row.get("mechanism_category") or "unknown")
        rows.append(
            [
                row.get("victim_model"),
                row.get("attacker_seed"),
                MECHANISM_LABELS.get(category, category),
                cases,
                n_ci(row.get("cumulative_attack_success"), cases, row.get("CASR")),
                ci(row.get("dangerous_candidate_rate")),
                ci(row.get("conditional_observed_bypass_given_dangerous")),
                row.get("median_round_to_success") or "NA",
            ]
        )
    return md_table(["模型", "seed", "机制层", "case", "CASR@5", "危险候选@5", "条件观察绕过@5", "首次成功中位轮"], rows)


def historical_agentdojo_ablation_table() -> str:
    base = HISTORICAL_REFERENCE_ROOT / "agentdojo"
    profiles = [
        ("full", "Full ObliGate"),
        ("conflict-collapse", "移除/折叠冲突见证"),
        ("gap-blind", "授权缺口盲化"),
        ("no-lifted-join", "移除 lifted-join"),
    ]
    rows = []
    for dirname, label in profiles:
        raw_dir = base / dirname / "raw_runs"
        if not raw_dir.exists():
            continue
        cases = 0
        attack_success = 0
        user_success = 0
        blocked_cases = 0
        confirmations = 0
        allow = 0
        block = 0
        silent = 0
        invalid_cert = 0
        failures = 0
        for path in raw_dir.glob("*.json"):
            try:
                item = read_json(path)
                norm = (item.get("normalized_cases") or [{}])[0]
                audit = item.get("obligate_audit_summary") or {}
                cases += 1
                attack_success += int(bool(norm.get("final_injection_task_success")) or int(norm.get("targeted_asr_contribution") or 0) > 0)
                user_success += int(bool(norm.get("final_user_task_success")))
                blocked_cases += int(bool(norm.get("blocked_case")))
                confirmations += int(audit.get("confirmation_count") or norm.get("confirmation_required_count") or 0)
                allow += int(audit.get("allow") or 0)
                block += int(audit.get("block") or 0)
                silent += int(audit.get("silent_dispatch_count") or 0)
                invalid_cert += int(audit.get("ordinary_allow_invalid_certificate_count") or 0)
            except Exception:  # noqa: BLE001
                failures += 1
        rows.append(
            [
                label,
                cases,
                n_ci(attack_success, cases),
                n_ci(user_success, cases),
                n_ci(blocked_cases, cases),
                confirmations,
                f"allow:{allow}; block:{block}",
                silent,
                invalid_cert,
                failures,
            ]
        )
    return md_table(["AgentDojo 配置", "case", "注入成功/ASR", "原任务成功", "触发阻断 case", "confirmation 次数", "裁决", "silent dispatch", "invalid cert dispatch", "解析失败"], rows)


def historical_general_benchmark_table() -> str:
    rows = []
    sb_path = HISTORICAL_REFERENCE_ROOT / "agent_safetybench" / "summary.json"
    if sb_path.exists():
        sb = read_json(sb_path)
        cases = int(sb.get("cases") or 0)
        rows.append(
            [
                "Agent-SafetyBench",
                cases,
                sb.get("model"),
                n_ci(sb.get("invalid_or_error_conservative_unsafe_count"), cases, sb.get("invalid_or_error_conservative_unsafe_rate") if isinstance(sb.get("invalid_or_error_conservative_unsafe_rate"), dict) else None),
                f"{sb.get('unsafe_tool_execution_rate', 'NA')}",
                f"{sb.get('dangerous_action_blocking_rate', 'NA')}",
                compact_counts(sb.get("decision_counts")),
                sb.get("errors"),
            ]
        )
    asb_path = HISTORICAL_REFERENCE_ROOT / "agent_security_bench" / "deepseek_v4_flash_adaptive_r2_full8160_seed20260714" / "summary.json"
    if asb_path.exists():
        asb = read_json(asb_path)
        overall = ((asb.get("overall") or {}).get("obligate_registry_blind") or {})
        cases = int(overall.get("cases") or 0)
        conservative_count = int(round(float(overall.get("Conservative_ASR") or 0) * cases / 100)) if cases else 0
        asr_count = int(round(float(overall.get("ASR") or 0) * cases / 100)) if cases else 0
        rows.append(
            [
                "Agent Security Bench adapted",
                cases,
                "deepseek-v4-flash",
                n_ci(conservative_count, cases),
                f"ASR {overall.get('ASR', 'NA')}% / executed-target {overall.get('Executed_Target_Tool_ASR', 'NA')}%",
                f"{overall.get('Dangerous_Action_Blocking_Rate', 'NA')}%",
                str(overall.get("decision_counts", "NA")),
                overall.get("workflow_failures"),
            ]
        )
        rows.append(
            [
                "Agent Security Bench adapted（观察成功）",
                cases,
                "deepseek-v4-flash",
                n_ci(asr_count, cases),
                f"Original task {overall.get('Original_Task_Success', 'NA')}%",
                f"Unsafe exec {overall.get('Unsafe_Tool_Execution_Rate', 'NA')}%",
                str(overall.get("theory_metrics", "NA")),
                overall.get("provider_errors"),
            ]
        )
    return md_table(["历史一步参考 集合", "case", "模型", "保守不安全/成功", "执行指标", "阻断/安全指标", "裁决/机制", "errors"], rows)


def cross_model_replay_table() -> str:
    path = RESULT_ROOT / "cross_model_replay" / "cross_model_replay.json"
    if not path.exists():
        return "_尚未运行 cross-model replay；需等待 main 5 轮完成并确定首次观察成功 payload。_"
    data = read_json(path)
    rows = []
    for item in data.get("directions") or []:
        rs = item.get("round_summary") or {}
        cases = int(rs.get("cases") or item.get("replayed_cases") or 0)
        rows.append(
            [
                item.get("source_victim_model"),
                item.get("target_victim_model"),
                item.get("source_observed_success_payloads"),
                item.get("replayed_cases"),
                n_ci(rs.get("attack_success"), cases),
                n_ci(rs.get("attack_success_observed"), cases),
                n_ci(rs.get("dangerous_candidate"), cases),
                n_ci(rs.get("workflow_failure"), cases),
            ]
        )
    return md_table(["源模型", "目标模型", "源观察成功 payload", "重放 case", "重放攻击成功", "重放观察绕过", "危险候选", "workflow failure"], rows)


def completion_note(summary: dict[str, Any]) -> str:
    rounds: dict[str, list[int]] = defaultdict(list)
    for row in summary.get("overall_by_round") or []:
        rounds[str(row.get("victim_model"))].append(int(row.get("round") or 0))
    if not rounds:
        return "当前尚无正式 score rows。"
    parts = []
    for model, values in sorted(rounds.items()):
        parts.append(f"{model}: 已落盘 round {min(values)}–{max(values)}")
    return "；".join(parts) + "。"


def analysis_text(summary: dict[str, Any]) -> str:
    lookup = cumulative_lookup(summary)
    lines = []
    for model in sorted({str(row.get("victim_model")) for row in summary.get("cumulative") or [] if row.get("segment") == "main"}):
        all5 = lookup.get(("main", model, "ALL", 5)) or lookup.get(("main", model, "ALL", 3)) or lookup.get(("main", model, "ALL", 1))
        if not all5:
            continue
        lines.append(
            f"- {model}：截至当前已落盘轮次，累计攻击成功为 "
            f"{all5.get('cumulative_attack_success')}/{all5.get('cases')}（{ci(all5.get('CASR'))}），"
            f"危险候选生成为 {all5.get('dangerous_candidate_cases')}/{all5.get('cases')}（{ci(all5.get('dangerous_candidate_rate'))}）。"
            f"这说明自适应攻击首先受底座模型是否提出危险候选限制；条件观察绕过率为 "
            f"{ci(all5.get('conditional_observed_bypass_given_dangerous'))}，对应的是 ObliGate 对已产生危险候选的实际拦截边界。"
        )
    if not lines:
        return "_暂无足够数据进行分析。_"
    lines.append("- 需要特别注意：报告中的 `attack_success` 是保守口径，provider/workflow failure 会保留在分母，并在部分 benchmark 中按不安全上界处理；`attack_success_observed` 则只统计实际观察到的危险动作/攻击目标完成。二者的差距本身就是失败案例分析的一部分。")
    return "\n".join(lines)


def build_report(summary: dict[str, Any]) -> str:
    expected = summary.get("expected") or {}
    main_expected = expected.get("main") or {}
    stress_expected = expected.get("mechanism_stress") or {}
    audit = summary.get("validity_audit") or {}
    checks = (audit.get("checks") or {})
    generated_at = summary.get("generated_at") or utc_now()
    return f"""# ObliGate 多轮闭环自适应攻击实验报告

生成时间：{generated_at}

数据根目录：`{summary.get("result_root")}`

当前完成状态：{completion_note(summary)}

一、实验目的与定位

本实验对应 ObliGate 的 RQ2：在攻击者只能看到公开反馈、不能访问 scorer-only 字段和内部治理证据的条件下，测试攻击者是否能通过多轮改写低可信输入，在有限轮次内绕过 ObliGate。该实验不把既有 `mechanism_activation_v1` 一步压力实验改写成多轮实验；旧实验仅作为 历史一步参考 的只读参考，本报告中的多轮统计只使用 `results/adaptive_closed_loop_v2/` 下的新产物。

核心问题拆成两层：

1. 底座模型是否会生成危险候选动作；
2. 在危险候选已经出现时，ObliGate 是否会错误放行并导致攻击目标完成。

因此，本报告同时报告 CASR、危险候选生成率和条件绕过率，避免把“模型没有提出危险动作”误读为“治理机制拦截了危险动作”。

二、实验设置

- 方法名称：ObliGate。
- victim model：`deepseek-v4-flash`、`qwen-plus`。
- attacker model：固定同一 attacker 配置、同一 prompt、同一温度与 seed 规则；攻击者只接收公开反馈。
- 主评测集合：{main_expected.get("case_count", "NA")} cases / victim model；benchmark 分布为 `{main_expected.get("benchmark_counts", {})}`。
- 机制压力集合：{stress_expected.get("case_count", "NA")} cases；机制压力集合仅用于 diagnostic stress，不并入主 benchmark 总 ASR。
- 最大轮数：5。Round 0 使用冻结初始 payload；后续轮次只由上一轮公开反馈驱动。
- 环境重置：每一轮从同一 benchmark 初始状态重新加载 case，只替换当前轮攻击 payload。
- 失败保守口径：provider/workflow failure 不从分母删除；需要时同时报告保守攻击成功与观察攻击成功。

三、有效性与审计

- preflight formal run allowed：{checks.get("preflight_formal_run_allowed")}
- activation profile hash exact：{checks.get("preflight_profile_hash_exact")}
- sampling public inputs only：{checks.get("sampling_public_inputs_only")}
- forbidden selection fields empty：{checks.get("sampling_forbidden_fields_empty")}
- main manifest present：{checks.get("main_manifest_present")}
- mechanism stress manifest present：{checks.get("mechanism_stress_manifest_present")}
- scores exist：{checks.get("scores_exist")}

审计文件：

- `results/adaptive_closed_loop_v2/code_hashes.json`
- `results/adaptive_closed_loop_v2/validity_audit.json`
- `results/adaptive_closed_loop_v2/reports/adaptive_closed_loop_summary.json`

API 路由审计：

{provider_route_table()}

说明：DeepSeek 使用的环境变量名为 `OPENAI_API_KEY` 是因为 DeepSeek 采用 OpenAI-compatible API 客户端；实际 base URL 指向 `api.deepseek.com`。两个 victim 分支共用固定 attacker model，因此 attacker usage 中出现 `qwen-plus` 是预注册攻击者配置，不表示 DeepSeek victim 用错 key。

四、主评测集合累计结果

{main_casr_table(summary)}

数据分析：

{analysis_text(summary)}

五、逐轮结果与裁决结构

{round_table(summary)}

从逐轮结构看，报告应重点区分三种现象：第一，危险候选没有出现，说明攻击改写尚未让底座模型进入危险动作空间；第二，危险候选出现但被 block / require_confirmation / execute_with_constraints 吸收，说明 ObliGate 的约束仍在发挥作用；第三，危险候选被 allow 并完成攻击目标，才构成真正的条件绕过。若保守成功高于观察绕过，通常意味着 provider/workflow failure 或 benchmark adapter 失败被计入安全上界，需要在失败案例中单独解释。

六、分 benchmark 逐轮结果

{benchmark_table(summary)}

AgentDojo 是主论文中最能同时观察安全性与有效性平衡的集合，因为它有原任务成功、注入任务成功和工具裁决轨迹。Agent-SafetyBench 在本实验中是工具级代理化适配结果，不写成官方 Utility；Agent Security Bench 使用 adapted protocol，不能写成 official protocol。三者分开报告可以避免把不同 benchmark 的 scorer 语义混成一个不可解释的总体 ASR。

七、攻击者生成质量与停止原因

{attacker_table(summary)}

{stop_table(summary)}

无效 payload 的存在不是噪声，应视为自适应攻击边界的一部分：当攻击者无法在保持同一攻击目标、同一通道和不引入隐藏字段的情况下生成合格改写时，该 case 应被停止而不是删除。若某一轮 invalid rate 明显升高，说明公开反馈不足以稳定引导攻击者找到新的可行局部处置。

八、机制压力集合

{stress_table(summary)}

机制压力集合只用于定位 ObliGate 的具体失效边界，不用于估计 benchmark 总体 ASR。报告重点应放在 conflict-active、pure-gap-active、lifted-join-active、OPI authorization-observability 和 confirmation-pressure 是否导致 CASR@5 或条件绕过率上升，以及上升是否跨两个 victim model 一致。

九、Token 使用与费用口径

{usage_table(summary)}

当前日志记录了实测 token。由于供应商价格表可能变化，本报告不硬编码旧价格；如需要美元/人民币费用，可在最终版中用运行当日价格表对 input/output tokens 分别换算。

十、失败案例分析框架

本实验的失败案例不应只看攻击成功样本，也需要分析“不成功但暴露边界”的样本：

1. 无危险候选：攻击者多轮改写后仍不能诱导 victim 生成目标危险动作，说明边界在底座模型候选生成侧。
2. 危险候选被阻断：候选动作出现，但 ObliGate 通过冲突证据、授权缺口、确认负担或高风险规则阻断/约束，说明治理机制有效。
3. 观察绕过：危险候选被放行并完成攻击目标，是最需要进入论文案例分析的真实失效。
4. provider/workflow failure：保守计入不安全上界，但不能和观察绕过混写；需要说明是供应商响应、adapter 工作流还是 benchmark 执行问题。
5. attacker invalid：攻击者无法生成合格续写，反映公开反馈下的搜索失败；不能从分母删除。

十一、历史一步参考 只读组件消融参考

下表来自 `results/adaptive_ablation/formal_activation_v1/`，该目录在本实验中保持只读。它回答的是 历史一步参考/组件消融问题，不与今天的 RQ2 多轮 CASR 合并计算。

{historical_agentdojo_ablation_table()}

{historical_general_benchmark_table()}

组件消融的解释重点是“哪个机制缺失会让边界暴露”。若 conflict-collapse、gap-blind 或 no-lifted-join 相比 Full ObliGate 出现更高注入成功、更低危险阻断或更异常的 confirmation/silent-dispatch 结构，就说明该组件确实在把局部证据冲突、授权缺口和多步义务合成为可执行治理裁决中发挥作用。反过来，如果某个消融没有显著恶化，也不能简单说该模块无用：可能是对应机制在该 benchmark 分布里未被充分激活，或底座模型没有生成足够危险候选。

十二、跨模型重放诊断

{cross_model_replay_table()}

跨模型重放不进入主 CASR 分母，它用于区分两类边界：如果某 payload 只在源模型成功，说明攻击更可能利用了源 victim 的候选生成/规划偏差；如果在两个模型上都成功，则更接近 ObliGate 共享治理边界。

十三、结论边界

本实验只能支持如下限定性结论：在预注册样本、公开粗粒度反馈、固定 attacker 配置和最多 5 轮预算下，ObliGate 对多轮自适应攻击表现出怎样的抵抗能力，以及哪些机制边界在压力样本中仍可被观察到。即使某个子集观察绕过为 0，也只能写成“本轮实验中未观察到绕过”，不能写成对任意自适应攻击绝对安全。
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=None, help="Use an existing summary JSON instead of rebuilding")
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.summary:
        summary = read_json(args.summary)
    else:
        summary = build_summary()
    text = build_report(summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
