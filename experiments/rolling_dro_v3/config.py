from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR.parent
@dataclass(frozen=True)
class Config:
    data_dir: Path = DEFAULT_DATA_DIR
    output_dir: Path = PROJECT_DIR / "outputs"
    cache_dir: Path = PROJECT_DIR / ".cache"
    target_service: float = 0.95
    holding_cost: float = 1.0
    shortage_cost: float | None = None
    calib_fraction: float = 0.80
    min_dro_shift: float = 0.005
    max_dro_shift: float = 0.0275
    confidence_beta: float = 0.05
    prior_weight: float = 3.0     # 在线水平更新的先验权重（越小=越快适应单品水平漂移）
    rolling_window: int = 6       # 在线水平更新窗口
    ewm_halflife: float = 2.0
    max_log_adjust: float = 2.0
    max_periods: int | None = None
    # ---- 大作业口径（PPT）：提前期、盘点周期、多周期 LP 设定 ----
    lead_time: int = 1          # 补货提前期 L（期）：第 t 期订货，第 t+L 期到货
    review_period: int = 0      # 盘点周期 r：保护期 = r + L（PPT 第 14 页）
                                # r = 0 → 连续盘点：只需覆盖提前期需求，保护期 = L = 1 期
    order_timing: str = "auto"  # 下单时点："auto"（r=0 取 post，否则 pre）| "pre" 期初 | "post" 期末
    n_scenarios: int = 200      # OR-tools 场景 LP 的样本场景数（经验分位分辨率 = 1/W；W=80 时 τ>0.9875 会饱和到场景最大值）
    service_metric: str = "instock"  # 服务水平口径：库存>0 的期数比例（PPT 口径）
    # ---- 多阶段随机规划（答疑 B8）+ 滚动时域（答疑 A0）----
    msp_horizon: int = 4        # 预测/决策窗口长度 H：答疑确认「预测 4 个周期，只执行第 1 期决策」
    msp_branches: int = 3       # 场景树每层分支数 N（场景总数 = N^H，节点总数 = Σ N^k）
    # ---- (s, S) 评分对照基线（答疑 A4）----
    ss_reorder_quantile: float = 0.90   # 再订货点 s 的分位（保护期需求）
    ss_target_quantile: float = 0.98    # 目标水位 S 的分位（保护期需求）
    def __post_init__(self) -> None:
        if self.order_timing not in ("auto", "pre", "post"):
            raise ValueError('order_timing must be one of "auto" / "pre" / "post"')
        if not 0.5 <= self.target_service < 1.0: raise ValueError("target_service must be in [0.5, 1.0)")
        if self.holding_cost <= 0: raise ValueError("holding_cost must be positive")
        if not 0.0 < self.calib_fraction < 1.0: raise ValueError("calib_fraction must lie in (0, 1)")
        if self.min_dro_shift < 0 or self.max_dro_shift < self.min_dro_shift: raise ValueError("invalid DRO shift bounds")
        if self.prior_weight < 0 or self.rolling_window < 1: raise ValueError("invalid rolling-update parameters")
        if self.max_periods is not None and self.max_periods < 1: raise ValueError("max_periods must be positive")
        if self.lead_time < 0 or self.review_period < 0: raise ValueError("lead_time and review_period must be >= 0")
        if self.review_period + self.lead_time < 1: raise ValueError("protection period r+L must be >= 1")
        if self.n_scenarios < 2: raise ValueError("n_scenarios must be >= 2")
        if self.msp_horizon < 2: raise ValueError("msp_horizon must be >= 2")
        if self.msp_branches < 2: raise ValueError("msp_branches must be >= 2")
        if not 0.0 < self.ss_reorder_quantile <= self.ss_target_quantile < 1.0:
            raise ValueError("require 0 < ss_reorder_quantile <= ss_target_quantile < 1")
        if 1.0 - 1.0 / self.n_scenarios < self.target_service:
            raise ValueError("n_scenarios too small: 经验分位分辨率 1/W 不足以区分目标服务水平，请增大 n_scenarios")
        if self.shortage_cost is None: object.__setattr__(self,"shortage_cost",self.holding_cost*self.target_service/(1.0-self.target_service))

    # ---- 由「盘点周期 r」派生的下单时点 ----
    # r = 0 表示连续盘点（每隔 0 期＝不间断盘点）：本期需求实现后立刻盘点下单，
    # 因此订单只需覆盖提前期 L 的需求；这与保护期 = r + L = L 完全自洽。
    # r ≥ 1 表示定期盘点：每期期初（需求实现前）下单，需覆盖 r + L 期。
    @property
    def post_demand(self) -> bool:
        if self.order_timing == "post": return True
        if self.order_timing == "pre": return False
        return int(self.review_period) == 0

    @property
    def protection_period(self) -> int:
        """保护期 = 盘点周期 r + 提前期 L（PPT 第 14 页公式）。"""
        return int(self.review_period) + int(self.lead_time)
