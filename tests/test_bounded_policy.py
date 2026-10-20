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
    torch.manual_seed(4); a=GRRLAgent(); obs=torch.randn(188)
    for idx in range(9):
        mask=torch.zeros(9,dtype=torch.bool);mask[idx]=True
        action,params,old_lp,_=a.select_action(obs,mask)
        lp,ent,_=a.actor_critic.evaluate_actions(obs[None],mask[None],torch.tensor([action]),params[None])
        assert torch.allclose(lp,torch.tensor([old_lp]),atol=1e-5)
        assert torch.allclose((lp-old_lp).exp(),torch.ones(1),atol=1e-5)
        assert torch.isfinite(ent).all()
        if idx<7: assert (params[:3]>0).all() and (params[:3]<1).all()


def test_sampling_and_density_describe_the_same_distribution():
    """Monte-Carlo: -E[log p(x)] == entropy and sample moments == analytic moments."""
    torch.manual_seed(0)
    for m, ls in [(.5, 0.), (.1, -1.), (.95, 1.), (.5, -3.)]:
        d = bounded_parameters(torch.tensor([m]), torch.tensor([ls]))
        x = d.sample((200000,))
        assert abs(float(-d.log_prob(x).mean()) - float(d.entropy())) < 0.02
        assert abs(float(x.mean()) - float(d.mean)) < 0.005
        assert abs(float(x.var()) - float(d.variance)) < 0.005
        assert float(d.entropy()) <= 1e-6  # differential entropy on a unit interval is <= 0 (uniform = 0)


def test_density_integrates_to_one_and_has_no_boundary_singularity():
    for m, ls in [(.02, 2.), (.98, 2.), (.5, -3.)]:
        d = bounded_parameters(torch.tensor(m), torch.tensor(ls))
        xs = torch.linspace(1e-6, 1 - 1e-6, 200001)
        area = torch.trapezoid(d.log_prob(xs).exp(), xs)
        assert abs(float(area) - 1.0) < 1e-3
        assert torch.isfinite(d.log_prob(xs)).all()


def test_deterministic_action_equals_head_mean_and_stays_in_range():
    a = GRRLAgent(); obs = torch.randn(188)
    mask = torch.ones(9, dtype=torch.bool); mask[7:] = False
    for _ in range(20):
        idx, params, lp, _ = a.select_action(obs, mask, deterministic=True)
        assert torch.isfinite(torch.tensor(lp))
        assert ((params >= 0) & (params <= 1)).all()


def test_mapped_physical_values_stay_inside_effect_ranges_at_the_boundaries():
    import yaml
    from dapn.dapn import DAPN
    from dataset.dapn_dataset import PARAM_SCHEMA
    from rl.grrl_agent import clamp_actions
    cfg = yaml.safe_load(open("config/effects_config.yaml"))["effects"]
    dapn = DAPN()
    for eff, names in PARAM_SCHEMA.items():
        for v in (0., 1.):
            x = clamp_actions(torch.full((len(names),), v))
            chain = {"chain": [{"effect": eff, "parameters": {n: float(x[i]) for i, n in enumerate(names)}}]}
            out = dapn.process(torch.randn(1, 4410) * .1, chain)
            assert torch.isfinite(out).all(), (eff, v)
            for i, n in enumerate(names):
                lo, hi = cfg[eff]["parameters"][n]["physical_range"]
                assert lo <= lo + float(x[i]) * (hi - lo) <= hi
