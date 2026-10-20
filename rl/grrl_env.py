"""Sequential FX editing MDP. Effects append; BACKTRACK removes the last effect.

BACKTRACK makes every warm-start prefix revisable: pop to the desired position,
then append new effects/parameters in a different order. It is an explicit policy
action, never an automatic search or optimizer. STOP alone finalizes a valid chain;
the operation budget is a time-limit truncation. Invalid actions raise ValueError
without mutation; DSP errors raise RuntimeError without committing the candidate.
"""
import copy
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from common.audio_pair import prepare_audio_pair
from common.config_loader import load_system_config, get_sample_rate
from common.losses import MultiResolutionSTFTLoss
from dapn.dapn import DAPN
from dataset.dapn_dataset import PARAM_SCHEMA, SUPPORTED_EFFECTS
from models.inverse_fx_net import InverseFXNet

EFFECT_ACTIONS=list(SUPPORTED_EFFECTS)+['STOP','BACKTRACK']
STOP_ACTION_IDX=7
BACKTRACK_ACTION_IDX=8
NUM_EFFECT_ACTIONS=9
MAX_PARAMS_PER_EFFECT=5
MAX_POSSIBLE_CHAIN_LENGTH=8


def improvement_percent(baseline, current):
    return 100.0*(baseline-current)/max(abs(baseline),1e-8)


class GRRLEnv:
    def __init__(self,predictor_checkpoint=None,device='cpu',max_chain_length=None,
                 min_chain_length=None,allow_repeated_effects=None,step_penalty=.05,
                 target_threshold_loss=.40,sample_rate=None,audio_duration_sec=2.,
                 pta_taste_dim=16,max_episode_steps=24,length_penalty=.02):
        self.device=torch.device(device)
        cfg=dict(load_system_config()['chain'])
        for name,value in [('max_chain_length',max_chain_length),('min_chain_length',min_chain_length),('allow_repeated_effects',allow_repeated_effects)]:
            if value is not None:cfg[name]=value
        self.max_chain_length=int(cfg['max_chain_length']);self.min_chain_length=int(cfg['min_chain_length'])
        self.allow_repeated_effects=bool(cfg['allow_repeated_effects'])
        effective_max=self.max_chain_length if self.allow_repeated_effects else min(7,self.max_chain_length)
        if not 0<=self.min_chain_length<=effective_max or not 1<=self.max_chain_length<=8:
            raise ValueError('Invalid chain limits (maximum supported history is 8)')
        if max_episode_steps<self.min_chain_length or max_episode_steps<1:raise ValueError('Invalid operation budget')
        self.max_episode_steps=max_episode_steps
        self.sample_rate=get_sample_rate()
        if sample_rate is not None and sample_rate!=self.sample_rate:raise ValueError('Environment/DAPN sample rates differ')
        self.audio_duration_sec=audio_duration_sec;self.max_samples=int(self.sample_rate*audio_duration_sec)
        self.step_penalty=step_penalty;self.length_penalty=length_penalty;self.target_threshold_loss=target_threshold_loss
        self.dapn=DAPN(chain_config={**cfg,'min_chain_length':0})  # partial prefixes can be empty
        self.loss_fn=MultiResolutionSTFTLoss()
        self.predictor=None
        if predictor_checkpoint is not None:
            self.predictor=InverseFXNet().to(self.device)
            self.predictor.load_checkpoint(predictor_checkpoint)
            self.predictor.eval()
        self.acoustic_dim=64;self.history_slots=8;self.history_dim=64;self.metrics_dim=4
        self.pta_taste_dim=pta_taste_dim;self.obs_dim=148+40  # 8 slots x 5 explicit parameter values
        if pta_taste_dim!=16:raise ValueError('Policy contract requires 16 taste dimensions')
        self.num_actions=NUM_EFFECT_ACTIONS;self.max_params_per_effect=5
        self.current_chain=[];self.current_audio=None;self.dry_audio=None;self.target_audio=None
        self.current_step=0;self.current_loss=0.;self.initial_loss=0.;self.start_loss=0.;self.predictor_loss=None
        self._done=True

    def reset(self,dry_audio,target_audio,use_warm_start=False,pta_taste=None):
        dry,wet=prepare_audio_pair(dry_audio,target_audio,device=self.device,sample_rate=self.sample_rate,duration=self.audio_duration_sec)
        if use_warm_start and self.predictor is None:raise ValueError('Warm-start requires an ordered predictor checkpoint')
        chain=[]
        with torch.no_grad():
            if use_warm_start:
                chain=self.predictor.predict_chain(dry,wet,min_length=self.min_chain_length,max_length=self.max_chain_length)['chain']
            current=self.dapn.process(dry,{'chain':chain}) if chain else dry.clone()
            if not torch.isfinite(current).all():raise RuntimeError('Nonfinite warm-start rendering')
            initial=float(self.loss_fn(dry,wet));start=float(self.loss_fn(current,wet))
        taste=torch.zeros(16,device=self.device) if pta_taste is None else pta_taste.to(self.device)
        if taste.shape!=(16,) or not torch.isfinite(taste).all():raise ValueError('Invalid taste vector')
        self.dry_audio,self.target_audio=dry,wet;self.current_audio=current;self.current_chain=copy.deepcopy(chain)
        self.initial_loss=initial;self.start_loss=start;self.predictor_loss=start if use_warm_start else None
        self.current_loss=start;self.current_step=0;self.pta_taste_vector=taste;self._done=False
        return self._get_observation(),dict(initial_loss=initial,dry_loss=initial,predictor_loss=self.predictor_loss,
            start_loss=start,current_loss=start,chain_length=len(chain),warm_started=use_warm_start,action_mask=self.get_action_mask())

    def get_action_mask(self):
        mask=torch.ones(9,dtype=torch.bool,device=self.device)
        if self._done:return torch.zeros_like(mask)
        n=len(self.current_chain)
        mask[7]=n>=self.min_chain_length;mask[8]=n>0
        if n>=self.max_chain_length:mask[:7]=False
        if not self.allow_repeated_effects:
            for step in self.current_chain:mask[SUPPORTED_EFFECTS.index(step['effect'])]=False
        # Reserve enough operations to return to the configured minimum after editing.
        remaining=self.max_episode_steps-self.current_step
        if n<=self.min_chain_length and remaining<=self.min_chain_length-n+1:mask[8]=False
        return mask

    def step(self,discrete_action,continuous_params=None):
        if self._done:raise RuntimeError('Episode is finished or uninitialized; call reset')
        raw=torch.as_tensor(discrete_action)
        if raw.numel()!=1 or not torch.isfinite(raw) or float(raw)!=int(raw):raise ValueError('Action must be one integer')
        idx=int(raw)
        if not 0<=idx<9 or not self.get_action_mask()[idx]:raise ValueError(f'Invalid or masked action: {idx}')
        prev=self.current_loss;candidate=copy.deepcopy(self.current_chain);old_len=len(candidate)
        if idx<7:
            effect=EFFECT_ACTIONS[idx];names=PARAM_SCHEMA[effect]
            p=torch.full((len(names),),.5,device=self.device) if continuous_params is None else torch.as_tensor(continuous_params,device=self.device)
            if p.ndim!=1 or len(p)<len(names) or not torch.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Parameters must be finite normalized values in [0,1]')
            candidate.append(dict(effect=effect,parameters={name:float(p[i]) for i,name in enumerate(names)}))
        elif idx==8:candidate.pop()
        # Transactional execution: any error leaves the committed chain/audio/state intact.
        with torch.no_grad():
            if idx==7:rendered=self.current_audio;new_loss=prev
            else:
                try:
                    rendered=self.dapn.process(self.dry_audio,{'chain':candidate}) if candidate else self.dry_audio.clone()
                    if not torch.isfinite(rendered).all():raise ValueError('Nonfinite DAPN output')
                    new_loss=float(self.loss_fn(rendered,self.target_audio))
                    if not math.isfinite(new_loss):raise ValueError('Nonfinite audio loss')
                except Exception as exc:raise RuntimeError(f'DAPN candidate rejected: {exc}') from exc
        self.current_step+=1;self.current_chain=candidate;self.current_audio=rendered;self.current_loss=new_loss
        target=new_loss<=self.target_threshold_loss and len(candidate)>=self.min_chain_length
        terminated=idx==7 or target
        truncated=not terminated and self.current_step>=self.max_episode_steps
        self._done=terminated or truncated
        # Potential difference for reconstruction/length, operation cost prevents edit cycles.
        components=dict(loss_improvement_reward=10*(prev-new_loss),step_penalty=0. if idx==7 else -self.step_penalty,
                        length_penalty=-self.length_penalty*(len(candidate)-old_len),
                        terminal_bonus=3. if target and self.start_loss>self.target_threshold_loss else 0.)
        reward=sum(components.values())
        info=dict(action=EFFECT_ACTIONS[idx],step=self.current_step,chain_length=len(candidate),prev_loss=prev,
                  current_loss=new_loss,delta_loss=prev-new_loss,reward=reward,reward_breakdown={**components,'total_step_reward':reward},
                  improvement_pct=improvement_percent(self.start_loss,new_loss),dry_improvement_pct=improvement_percent(self.initial_loss,new_loss),
                  start_loss=self.start_loss,predictor_loss=self.predictor_loss,target_achieved=target,action_mask=self.get_action_mask(),
                  terminated=terminated,truncated=truncated)
        return self._get_observation(),reward,terminated,truncated,info

    def _get_observation(self):
        with torch.no_grad():
            win=torch.hann_window(512,device=self.device)
            def spec(x):return torch.stft(x.reshape(1,-1),512,256,window=win,return_complex=True).abs()
            diff=(spec(self.target_audio)-spec(self.current_audio)).mean(-1)
            acoustic=F.adaptive_avg_pool1d(diff[None],64).flatten()
            history=torch.zeros(8,8,device=self.device);history[:,0]=1
            params=torch.zeros(8,5,device=self.device)
            for i,step in enumerate(self.current_chain):
                history[i,0]=0;history[i,SUPPORTED_EFFECTS.index(step['effect'])+1]=1
                for j,k in enumerate(PARAM_SCHEMA[step['effect']]):params[i,j]=step['parameters'][k]
            metrics=torch.tensor([self.current_step/self.max_episode_steps,len(self.current_chain)/self.max_chain_length,
                self.current_loss/5.,improvement_percent(self.start_loss,self.current_loss)/100],device=self.device)
            return torch.cat([acoustic,history.flatten(),metrics,self.pta_taste_vector,params.flatten()])

    def get_current_chain_summary(self):
        return dict(chain=copy.deepcopy(self.current_chain),loss=self.current_loss,num_effects=len(self.current_chain),sequence=[s['effect'] for s in self.current_chain])
