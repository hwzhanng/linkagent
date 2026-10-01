"""Checksum-guarded reversible installation. Default is dry-run; --apply writes candidate files.
This does not enable the candidate or certify Isaac contact behavior.
"""
import argparse,datetime,hashlib,json,pathlib,shutil
R=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--repo',required=True,type=pathlib.Path);a.add_argument('--apply',action='store_true');a.add_argument('--rollback',type=pathlib.Path);args=a.parse_args();repo=args.repo.resolve();m=json.loads((R/'install_manifest.json').read_text())
 if args.rollback:
  backup=args.rollback.resolve();state=json.loads((backup/'rollback.json').read_text());assert pathlib.Path(state['repo'])==repo
  for rel,digest in state['installed'].items():
   if not (repo/rel).is_file() or sha(repo/rel)!=digest:raise SystemExit(f'Rollback refuses modified/missing file: {rel}')
  print('Rollback checked',len(state['installed']),'files')
  if not args.apply:return
  for rel in state['installed']:
   if rel in state['existed']:shutil.copy2(backup/rel,repo/rel)
   else:(repo/rel).unlink()
  print('Restored previous checkout files; backup retained.');return
 for rel,accepted in m['baseline_sha256'].items():
  if not (repo/rel).is_file() or sha(repo/rel) not in accepted:raise SystemExit(f'Baseline mismatch: {rel}; no files changed')
 for rel,digest in m['patch_sha256'].items():
  if sha(R/'patch'/rel)!=digest:raise SystemExit(f'Corrupt patch: {rel}')
  if (repo/rel).exists() and rel not in m['baseline_sha256'] and sha(repo/rel)!=digest:raise SystemExit(f'Refusing to overwrite unrelated candidate file: {rel}')
 print('Source and patch checks passed. Candidate files:',len(m['patch_sha256']))
 print('USD mesh dependencies are enforced at startup; missing meshes prevent candidate launch.')
 if not args.apply:print('Dry run only; use --apply to install.');return
 backup=repo/'physical_prior_backups'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');backup.mkdir(parents=True);state={'repo':str(repo),'existed':[],'installed':m['patch_sha256']}
 for rel in m['patch_sha256']:
  src=R/'patch'/rel;dst=repo/rel
  if dst.exists():p=backup/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dst,p);state['existed'].append(rel)
  dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
 (backup/'rollback.json').write_text(json.dumps(state,indent=2));print('Installed with rollback backup:',backup)
 print('Candidate opt-in: LINKAGENT_PHYSICAL_PRIOR=1 for new Physics-v1/v2 qualification runs only.')
if __name__=='__main__':main()
