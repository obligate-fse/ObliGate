"""Flat deterministic decision tables, no EOC/ECP/join/certificate calls.

Policy predicates are evaluated independently on predeclared concrete controls.
Single restricts executable controls to existing individual-rule remedies.
Catalog adds all predeclared policy controls as a stronger sensitivity control.
Neither constructs a new joint remedy or issues/verifies action certificates.
"""
from obligate.theory.model import exec_semantics,is_authorization_predicate
from obligate.theory.runtime import requirements_for_action
from obligate.theory.witness import BipolarWitnessEngine

def at_least(plan,lower):
    return exec_semantics(plan)<=exec_semantics(lower) and plan.audit.no_weaker_than(lower.audit)

def predicate_pass(plan,o):
    orders={'execution_env':['host','sandbox','no_execute'],'network_scope':['allow','allowlist','deny'],'data_scope':['raw','redact','no_sensitive']}
    if not plan.audit.consistent:return False
    if o.forbid_dispatch and not plan.block_like:return False
    if o.forbid_confirmation and plan.confirmation:return False
    if o.require_approval and plan.human_gate!='approval_required' and not plan.block_like:return False
    if not o.audit_must<=plan.audit.must:return False
    for key,order in orders.items():
        minimum=getattr(o,'min_'+key)
        if minimum is not None and order.index(getattr(plan,key))<order.index(minimum):return False
    return True

class FlatGate:
    def __init__(self,policy,realizer,mode):
        if mode not in ['single','catalog']:raise ValueError(mode)
        self.policy=policy;self.mode=mode
        all_controls=tuple(realizer.materialize(r) for r in policy.realizations)
        atomic={policy.default_plan.digest}|{r.plan.digest for r in policy.remedies}|{r.plan.digest for r in policy.fallback_remedies}
        self.controls=tuple(r for r in all_controls if mode=='catalog' or r.plan.block_like or r.plan.digest in atomic)
        self.rule_table={}
        for template in policy.triggers:
            remedies=[r.plan for r in policy.remedies if r.trigger_id==template.trigger_id]
            allowed=set()
            for r in self.controls:
                if not r.available:continue
                if not r.plan.audit.must<=r.actual_audit<=r.plan.audit.allowed:continue
                required={x.capability for x in template.implementation_constraints}
                if r.plan.block_like:required&={'audit','state_store'}
                if not required<=r.capabilities:continue
                if not all(predicate_pass(r.plan,o) for o in template.semantic_obligations):continue
                if not r.plan.block_like and not any(at_least(r.plan,p) for p in remedies):continue
                allowed.add(r.realization_id)
            self.rule_table[(template.kind,template.subject)]=allowed

    def decide(self,action,evidence,*,witness_limit=None):
        # Same signed facts, trust filter and grounded inference rules as Full.
        assertions=tuple(x for x in evidence.assertions if not (x.atom.polarity=='+' and is_authorization_predicate(x.atom.predicate) and not x.trusted_for_authorization))
        closure=BipolarWitnessEngine(witness_limit=witness_limit).close(assertions,self.policy.rules)
        requirements=requirements_for_action(action);keys=[]
        for atom in requirements:
            if closure.bipolar(atom.predicate,atom.arguments).value!='support-only':
                subject=self.policy.runtime_claim_map.get(atom.predicate)
                keys.append(('gap',subject) if subject is not None else ('hazard',self.policy.runtime_hazard_map['fallback']))
            if atom.unsigned_key in evidence.contract.deny:keys.append(('hazard','explicit_deny'))
        overflow={atom.unsigned_key for atom in requirements if closure.overflow_affects(atom)}|{key[1:] for key in closure.overflow}
        for subject in overflow:keys.append(('overflow',self.policy.runtime_claim_map.get(subject.split('(')[0],'authorized')))
        for signal in evidence.hazards:
            control='hard' if signal.hard else signal.preferred_control
            keys.append(('hazard',self.policy.runtime_hazard_map.get(control,self.policy.runtime_hazard_map['fallback'])))
        keys.extend(('hazard','explicit_deny') for _ in evidence.explicit_denies)
        eligible=[]
        for r in self.controls:
            if not r.available or not r.plan.audit.consistent:continue
            if not r.plan.audit.must<=r.actual_audit<=r.plan.audit.allowed:continue
            if any(not root.startswith(('trigger:','backend:','baseline:')) for _,root in r.control_roots):continue
            if any(r.realization_id not in self.rule_table.get(k,set()) for k in keys):continue
            if not r.plan.block_like and not at_least(r.plan,self.policy.default_plan):continue
            eligible.append(r)
        nonblock=[r for r in eligible if not r.plan.block_like]
        selected=min(nonblock or eligible,key=lambda r:(*r.utility_vector,r.realization_id),default=None)
        if selected is None:outcome='block_with_compliance_error'
        elif selected.plan.block_like:outcome='block'
        elif selected.plan.confirmation and evidence.confirmation is None:outcome='require_confirmation'
        elif selected.plan==self.policy.default_plan and all(root=='baseline' or root.startswith('baseline:') for _,root in selected.control_roots):outcome='allow'
        else:outcome='execute_with_constraints'
        return {'decision':outcome,'execute':outcome in ['allow','execute_with_constraints'],'selected':selected,
                'active_rules':sorted(set(keys)),'closure':closure.to_dict(),'certificate':None}
