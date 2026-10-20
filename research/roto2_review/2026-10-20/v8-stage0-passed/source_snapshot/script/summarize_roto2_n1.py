"""Summarize N-1 outcomes without treating candidate count as feasibility."""
from pathlib import Path
import argparse,json,hashlib


def summarize(path):
    source=path.read_bytes();d=json.loads(source);rows=d['results'];eligible=[r for r in rows if r['baseline_ok']]
    def all_trials(r,key,include_control=True):
        return all(t[key] for t in r['trials'][0 if include_control else 1:])
    groups={}
    for index in range(6):
        fields=['actually_released','stable_through_release_and_recovery','four_pad_recovered','physical_failure']
        name=rows[0]['trials'][index]['condition'] if rows else str(index)
        groups[name]={'denominator_baseline_eligible':len(eligible),**{key:sum(r['trials'][index][key] for r in eligible) for key in fields}}
    return dict(source=str(path),source_sha256=hashlib.sha256(source).hexdigest(),
        completed=d['validation_completed'],candidates=len(rows),baseline_eligible=len(eligible),
        unique_source_grasps=len({str(r.get('source_grasp_id',r['candidate_id'])) for r in rows}),
        accepted=sum(r['n_minus_one_passed'] for r in rows),per_condition=groups,
        all_actual_releases=sum(all_trials(r,'actually_released') for r in eligible),
        all_stable_releases_and_recoveries=sum(all_trials(r,'stable_through_release_and_recovery') for r in eligible),
        all_four_pad_recoveries=sum(all_trials(r,'four_pad_recovered') for r in eligible),
        all_released_without_physical_failure=sum(all_trials(r,'actually_released') and not any(t['physical_failure'] for t in r['trials']) for r in eligible),
        note='Counts are tested configurations; release-profile variants may share source_grasp_id. Not independent learned policies. No overall stage0 pass from this summary; frozen library, independent conditions, long hold, and video remain required.')


def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();r=summarize(a.input)
    text=json.dumps(r,indent=2,ensure_ascii=False)+'\n'
    if a.output:a.output.write_text(text)
    print(text)

if __name__=='__main__':main()
