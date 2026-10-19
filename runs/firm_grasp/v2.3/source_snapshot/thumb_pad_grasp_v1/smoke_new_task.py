"""独立新任务的真实维度/缓存/接触守卫接缝检查，不训练。"""
import argparse
import json
from pathlib import Path
import traceback
from isaaclab.app import AppLauncher


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--num-envs',type=int,default=1)
    parser.add_argument('--steps',type=int,default=30)
    parser.add_argument('--actions-json',default=None,help='显式逐环境动作表，仅用于有限物理可达性探针')
    parser.add_argument('--thumb-mapping',choices=['legacy','independent'],default='legacy')
    parser.add_argument('--thumb-first',action='store_true')
    parser.add_argument('--center-local-mm',nargs=3,type=float,default=[-79.16,11.16,-160.])
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    launcher=AppLauncher(args);env=None
    try:
        import torch
        from .env import ThumbPadGraspEnv,make_config,ACTOR_DIM,CRITIC_DIM
        env=ThumbPadGraspEnv(cfg=make_config(num_envs=args.num_envs,device=args.device,seed=42,
            thumb_mapping=args.thumb_mapping,stage='contact',pregrasp_distance_m=.005,fruit_center_local_m=tuple(x/1000 for x in args.center_local_mm)))
        env.pad_trace_enabled=True
        obs,_=env.reset()
        initial_q=env._robot.data.joint_pos.torch.tolist()
        initial_bodies=env._robot.data.body_pose_w.torch.tolist()
        initial_joint_names=list(env._robot.joint_names)
        initial_body_names=list(env._robot.body_names)
        initial_fruit=env._tomato.data.root_pose_w.torch.tolist()
        assert obs['policy'].shape==(args.num_envs,ACTOR_DIM)
        assert obs['critic'].shape==(args.num_envs,CRITIC_DIM)
        action=torch.tensor([.6,.5,0,0,.1,0,0,0,0,0,.8,.8,.8,.8],device=env.device).repeat(args.num_envs,1)
        if args.actions_json is not None:
            action=torch.tensor(json.loads(args.actions_json),device=env.device)
            if action.shape!=(args.num_envs,14) or not bool(torch.isfinite(action).all()) or bool((action.abs()>1).any()):
                raise ValueError('动作表必须为有限、有界的num_envs×14数组')
        with (args.output/'physical_substeps.jsonl').open('x') as stream:
            for step in range(args.steps):
                commanded=action*min(1.,step/35)
                if args.thumb_first:
                    age=env.episode_length_buf.float()
                    commanded=action*((age-20)/25).clamp(0,1)[:,None]
                    thumb_slice=slice(2,6) if args.thumb_mapping=="independent" else slice(4,6)
                    commanded[:,thumb_slice]=action[:,thumb_slice]*(age/15).clamp(0,1)[:,None]
                obs,reward,terminated,truncated,extras=env.step(commanded)
                assert torch.isfinite(reward).all() and torch.isfinite(obs['policy']).all()
                for record in env.drain_pad_records():stream.write(json.dumps(record,allow_nan=False)+'\n')
        (args.output/'result.json').write_text(json.dumps({'status':'RUNTIME_SMOKE_COMPLETED',
            'not_training':True,'not_success_acceptance':True,'policy_shape':list(obs['policy'].shape),
            'critic_shape':list(obs['critic'].shape),'task_identity':env.task_identity(),
            'declared_actions':action.tolist(),'thumb_first':args.thumb_first,
            'collision_aabb':{'body_names':[env._robot.body_names[i] for i in env._hand_collision_body_ids],
                'centers':env._hand_collision_aabb_centers.tolist(),'half_extents':env._hand_collision_aabb_half_extents.tolist()},
            'initial_q':initial_q,'initial_body_poses_w':initial_bodies,
            'initial_joint_names':initial_joint_names,'initial_body_names':initial_body_names,'initial_fruit_pose_w':initial_fruit,'first_reset_poses':env.first_reset_poses,
            'completed_episodes':env.completed_episodes,'last_contact':{k:v.tolist() for k,v in env.pad_observer.latest.items()}},indent=2,default=str))
    except BaseException:
        (args.output/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        if env is not None:env.close()
        launcher.app.close()


if __name__=='__main__':main()
