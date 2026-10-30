"""Both policies retain the pinned fused-v2 frontend, clone, codec and barriers."""
import ablation4_sources
from common import replace_once


def adapter_source():
    return ablation4_sources.adapter_source()


def taco_source():
    return ablation4_sources.taco_source()


def worker_source(source_only=False):
    source = ablation4_sources.worker_source(source_only)
    source = replace_once(source, "os.environ.get('H100_ABLATION4_EXECUTE')",
                          "os.environ.get('H100_ARRIVAL_OP2_EXECUTE')")
    source = replace_once(source, 'GPU worker requires ablation4 --execute',
                          'GPU worker requires arrival_op2 --execute')
    if source_only:
        source = replace_once(source, 'Source-only scope worker is permanently non-executable',
                              'Source-only arrival_op2 worker is permanently non-executable')
    anchor = "PLAN = json.loads((HERE / 'config.json').read_text())"
    source = replace_once(source, anchor, anchor+'''
from routing import PROTOCOL, execution_timing
execution = execution_timing()
if (PLAN.get('protocol') != PROTOCOL or PLAN.get('execution_timing') != execution or
        any(type(PLAN.get(key)) is not int or PLAN[key] != value for key, value in execution.items())):
    raise ValueError('Arrival-op2 requires frozen 10-warmup / 50-timed-step execution timing')
'''.rstrip())
    source = replace_once(source, "POLICY = os.environ['E2E_POLICY']", "POLICY = os.environ['E2E_POLICY']"+'''
from routing import A, S, IDENTITY_SPANS
schedule = PLAN.get('schedule_options')
if (not isinstance(schedule, dict) or set(schedule) != {'attn', 'mlp'} or
        any(type(value) is not int or value not in (1, 4, 8, 16, 32) for value in schedule.values())):
    raise ValueError('Missing or invalid frozen scheduling selection')
expected_spans = IDENTITY_SPANS if POLICY == A else schedule
if POLICY not in (A, S) or any(os.environ.get('H100_OP2_'+role.upper()+'_SPAN') != str(span)
                              for role, span in expected_spans.items()):
    raise ValueError('Runtime scheduling environment differs from the frozen selection')
'''.rstrip())
    return replace_once(source, "report['protocol'] = PLAN['protocol']",
                         "report['execution_timing'] = execution\n"
                         "report['schedule_options'] = schedule\n"
                         "report['runtime_spans'] = expected_spans\nreport['protocol'] = PLAN['protocol']")
