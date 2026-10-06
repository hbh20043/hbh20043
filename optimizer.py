import numpy as np
import pandas as pd
import json
from .models import ModelUnavailable
from functools import lru_cache
from .profiling import timed,section

@lru_cache(maxsize=1)#这个函数第一次运行以后，把结果记住；以后再调用，就直接返回上次的结果，不重新执行
def _installed_solvers():
    import cvxpy as cp
    return tuple(cp.installed_solvers())

@timed('optimizer.constraint_audit')
def _constraint_violation(constraints):
    return max(float(np.max(np.asarray(c.violation()))) for c in constraints)#找出最大的违反约束值

@timed('optimizer.build_problem')
def optimize(alpha, wb, pretrade, sigma_daily, exposures, beta, adv, nav, H, no_alpha, frozen, cfg):
    import cvxpy as cp
    axes = alpha.index
    for obj in (wb,pretrade,sigma_daily,exposures,beta,adv,no_alpha,frozen):
        if not axes.equals(obj.index): raise ValueError("ASSET_ORDER_MISMATCH")
    if not axes.equals(sigma_daily.columns): raise ValueError("COVARIANCE_ORDER_MISMATCH")
    if not np.isfinite(sigma_daily).all().all():
        raise ModelUnavailable("INVALID_OPTIMIZER_INPUT",details={"input":"sigma_daily"})
    invalid_adv = adv.lt(0) | (adv.notna() & ~np.isfinite(adv))
    if invalid_adv.any():
        raise ModelUnavailable("INVALID_ADV_INPUT",details={"security_ids":list(axes[invalid_adv])})
    if cfg.missing_adv_rule != "FREEZE_POSITION_NO_CAPACITY":
        raise ValueError("Unsupported missing ADV rule")
    # Missing history is not evidence of liquidity. Freeze only these positions;
    # keep the complete-window ADV definition and all portfolio risk constraints.
    unavailable_adv = adv.isna() | adv.eq(0)
    frozen = frozen | unavailable_adv
    capacity_adv = adv.mask(unavailable_adv,0.)
    liquidity_diag = dict(missing_adv_rule=cfg.missing_adv_rule,
        adv_observation_rule=cfg.adv_observation_rule,
        unavailable_adv_count=int(unavailable_adv.sum()),
        unavailable_adv_security_ids=list(axes[unavailable_adv]),
        unavailable_adv_pretrade_weight=float(pretrade.loc[unavailable_adv].sum()),
        unavailable_adv_benchmark_weight=float(wb.loc[unavailable_adv].sum()))
    if cfg.solver not in _installed_solvers(): raise RuntimeError("ENVIRONMENT_ERROR: CLARABEL is required")
    # Actual frozen holdings take precedence: exact benchmark matching is only
    # possible for unfrozen names. Keep the full benchmark in every risk bound.
    frozen_no_prediction=no_alpha & wb.gt(0) & frozen
    exact_mask=no_alpha & wb.gt(0) & ~frozen if cfg.exact_no_prediction_benchmark() else pd.Series(False,index=axes)
    liquidity_diag.update(frozen_no_prediction_weight_rule='KEEP_PRETRADE',
        frozen_no_prediction_count=int(frozen_no_prediction.sum()),
        frozen_no_prediction_security_ids=list(axes[frozen_no_prediction]),
        frozen_no_prediction_active_weights=json.dumps((pretrade-wb).loc[frozen_no_prediction].to_dict(),ensure_ascii=False),
        frozen_no_prediction_max_active_weight=float((pretrade-wb).loc[frozen_no_prediction].abs().max()) if frozen_no_prediction.any() else 0.)
    n = len(axes)
    w = cp.Variable(n)
    h = w-wb.to_numpy()
    delta = w-pretrade.to_numpy()
    sigma = sigma_daily.to_numpy()*H
    constraints = [w>=0, cp.sum(w)==1-cfg.cash_target_weight,
                   h<=cfg.stock_active_bound, h>=-cfg.stock_active_bound,
                   cp.quad_form(h,cp.psd_wrap(sigma))<=cfg.annual_te_limit**2*H/cfg.annualization_days,
                   cp.abs(delta)<=cfg.adv_participation_rate*cfg.execution_adv(capacity_adv.to_numpy())/nav,
                   cp.abs(beta.to_numpy()@h)<=cfg.beta_bound]
    for cols, bound in (([c for c in exposures if c.startswith("industry_")],cfg.industry_bound),
                        ([c for c in exposures if not c.startswith("industry_")],cfg.style_bound)):
        if cols: constraints.append(cp.abs(exposures[cols].to_numpy().T@h)<=bound)
    for i in range(n):
        if exact_mask.iloc[i]:
            # Unfrozen target equality does not bypass ADV or portfolio risk.
            constraints.append(w[i]==wb.iloc[i])
        if frozen.iloc[i]: constraints.append(w[i]==pretrade.iloc[i])
        elif wb.iloc[i] == 0: constraints.append(w[i]==0)
        elif no_alpha.iloc[i] and not exact_mask.iloc[i]: constraints.append(w[i]==(1-cfg.cash_target_weight)*wb.iloc[i])
    # Linear one-way commission/slippage estimate; minimum charges are charged in execution.
    cost = (cfg.commission+cfg.slippage)*cp.norm1(delta)
    if cfg.alpha_transform=='bl':
        # alpha carries full mu_BL in this explicit contract; same delta as Pi.
        objective = cp.Maximize(alpha.to_numpy()@w-cfg.risk_aversion/2*cp.quad_form(w,cp.psd_wrap(sigma))-cfg.cost_coefficient*cost)
    else:
        objective = cp.Maximize(alpha.to_numpy()@h-cfg.risk_aversion/2*cp.quad_form(h,cp.psd_wrap(sigma))-cfg.cost_coefficient*cost)
    problem = cp.Problem(objective,constraints)
    with section('optimizer.solve'):
        try:
            problem.solve(solver="CLARABEL",tol_gap_abs=1e-10,tol_feas=1e-10,tol_gap_rel=1e-10,max_iter=300)
        except cp.error.SolverError as error:
            # A solver error is not proof of infeasibility. Keep it distinct in
            # the period audit and use the engine's existing failure policy.
            raise ModelUnavailable('OPTIMIZER:solver_error',details=dict(liquidity_diag,
                solver='CLARABEL',solver_error=str(error))) from error
    if problem.status != cp.OPTIMAL:
        raise ModelUnavailable(f"OPTIMIZER:{problem.status}",details=liquidity_diag)
    violation = _constraint_violation(constraints)
    if violation > cfg.constraint_tolerance: raise ModelUnavailable(f"CONSTRAINT_VIOLATION:{violation}")
    # Solver residuals on an exact zero equality must never become a new buy
    # in a former constituent. Frozen legacy positions keep their own equality.
    exact_zero = wb.eq(0).to_numpy() & ~frozen.to_numpy(dtype=bool)
    projected = np.asarray(w.value).copy()
    zero_residual = float(np.max(np.abs(projected[exact_zero]))) if exact_zero.any() else 0.
    projected[exact_zero] = 0.
    projected[exact_mask.to_numpy(dtype=bool)] = wb.loc[exact_mask].to_numpy()
    projected[frozen.to_numpy(dtype=bool)] = pretrade.loc[frozen].to_numpy()
    w.value = projected
    violation = _constraint_violation(constraints)
    if violation > cfg.constraint_tolerance: raise ModelUnavailable(f"PROJECTED_CONSTRAINT_VIOLATION:{violation}")
    target = pd.Series(w.value,index=axes)
    active = target-wb
    return target, dict(solver_status=problem.status,target_constraint_violation=violation,
                        predicted_te=float(np.sqrt(max(active@ sigma_daily @active,0)*cfg.annualization_days)),
                        objective=float(problem.value),objective_contract='TOTAL_RETURN_TOTAL_RISK' if cfg.alpha_transform=='bl' else 'ACTIVE_ALPHA_ACTIVE_RISK',
                        target_turnover=float(np.abs(target-pretrade).sum()),
                        target_cash=cfg.cash_target_weight,legacy_zero_residual=zero_residual,
                        no_prediction_fixed_count=int(exact_mask.sum()),
                        no_prediction_max_active_weight=float(active.loc[exact_mask].abs().max()) if exact_mask.any() else 0.,
                        no_prediction_weight_rule='MATCH_BENCHMARK' if cfg.exact_no_prediction_benchmark() else 'CASH_SCALED_BENCHMARK',
                        no_specific_prediction_fixed_count=int(exact_mask.sum()) if cfg.bl_branch=='specific' else 0,
                        no_specific_prediction_max_active_weight=float(active.loc[exact_mask].abs().max()) if exact_mask.any() and cfg.bl_branch=='specific' else 0.,**liquidity_diag)
