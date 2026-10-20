import torch
from rl.grrl_agent import GRRLAgent, bounded_parameters


def test_beta_density_entropy_and_boundary_gradients():
    m=torch.tensor([.2,.5,.8],requires_grad=True); ls=torch.zeros(3,requires_grad=True)
    d=bounded_parameters(m,ls)
    x=torch.tensor([1e-6,.5,1-1e-6]); lp=d.log_prob(x)
    assert torch.isfinite(lp).all() and torch.isfinite(d.entropy()).all()
    (lp.sum()+d.entropy().sum()).backward()
    assert torch.isfinite(m.grad).all() and torch.isfinite(ls.grad).all()
    draws=d.sample((2000,))
    assert ((draws>0)&(draws<1)).all()


def test_sampling_evaluation_and_importance_ratio_agree():
    torch.manual_seed(4); a=GRRLAgent(); obs=torch.randn(148)
    for idx in range(8):
        mask=torch.zeros(8,dtype=torch.bool);mask[idx]=True
        action,params,old_lp,_=a.select_action(obs,mask)
        lp,ent,_=a.actor_critic.evaluate_actions(obs[None],mask[None],torch.tensor([action]),params[None])
        assert torch.allclose(lp,torch.tensor([old_lp]),atol=1e-5)
        assert torch.allclose((lp-old_lp).exp(),torch.ones(1),atol=1e-5)
        assert torch.isfinite(ent).all()
        if idx<7: assert (params[:3]>0).all() and (params[:3]<1).all()
