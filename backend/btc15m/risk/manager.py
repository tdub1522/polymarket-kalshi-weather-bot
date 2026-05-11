"""Risk manager: gate checks and Kelly position sizing for KXBTC15M signals."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import List, Tuple


class RiskManager:
    YES_PRICE_CAP             = 0.72
    NO_PRICE_CAP              = 0.65
    BLOCKED_UTC_HOURS         = {8, 11, 16, 18, 21}
    MIN_STRIKE_DISTANCE_PCT   = 0.0002
    DAILY_LOSS_CAP            = 150.0
    MAX_TRADES_PER_DAY        = 48
    MAX_POSITION_PCT          = 0.20
    MIN_ENTRY_WINDOW_SECONDS  = 360   # 6 min
    MAX_ENTRY_WINDOW_SECONDS  = 540   # 9 min
    GOLDEN_ZONE_LOW           = 0.65
    GOLDEN_ZONE_HIGH          = 0.73
    GOLDEN_ZONE_MAX_SECONDS   = 720   # 12 min

    def __init__(self, daily_trades: int = 0, daily_loss: float = 0.0) -> None:
        self._daily_trades = daily_trades
        self._daily_loss   = daily_loss

    def check_all_gates(
        self,
        market,
        spot: float,
        recommendation: str,
        yes_ask: float,
        no_ask: float,
        bankroll: float,
    ) -> Tuple[bool, List[str]]:
        """Run all pre-trade gates in order. Return (passed, [failed reasons])."""
        reasons: List[str] = []

        # 1. Blocked UTC hours
        utc_hour = datetime.now(timezone.utc).hour
        if utc_hour in self.BLOCKED_UTC_HOURS:
            reasons.append(f"blocked UTC hour {utc_hour}")

        # 2. Timing window
        secs = market.seconds_to_expiry
        in_golden = self.GOLDEN_ZONE_LOW <= yes_ask <= self.GOLDEN_ZONE_HIGH
        if in_golden:
            if not (180 <= secs <= self.GOLDEN_ZONE_MAX_SECONDS):
                reasons.append(
                    f"timing {secs:.0f}s outside golden-zone window [180, {self.GOLDEN_ZONE_MAX_SECONDS}]"
                )
        else:
            if not (self.MIN_ENTRY_WINDOW_SECONDS <= secs <= self.MAX_ENTRY_WINDOW_SECONDS):
                reasons.append(
                    f"timing {secs:.0f}s outside window "
                    f"[{self.MIN_ENTRY_WINDOW_SECONDS}, {self.MAX_ENTRY_WINDOW_SECONDS}]"
                )

        # 3. Strike distance from spot
        if spot > 0:
            dist_pct = abs(market.floor_strike - spot) / spot
            if dist_pct < self.MIN_STRIKE_DISTANCE_PCT:
                reasons.append(
                    f"strike too close to spot: {dist_pct:.4%} < {self.MIN_STRIKE_DISTANCE_PCT:.4%}"
                )

        # 4/5. Price caps
        if recommendation == "BUY_YES" and yes_ask > self.YES_PRICE_CAP:
            reasons.append(f"yes_ask {yes_ask:.2f} > cap {self.YES_PRICE_CAP}")
        if recommendation == "BUY_NO" and no_ask > self.NO_PRICE_CAP:
            reasons.append(f"no_ask {no_ask:.2f} > cap {self.NO_PRICE_CAP}")

        # 6. Daily trade count
        if self._daily_trades >= self.MAX_TRADES_PER_DAY:
            reasons.append(f"daily trades {self._daily_trades} >= {self.MAX_TRADES_PER_DAY}")

        # 7. Daily loss cap
        if self._daily_loss >= self.DAILY_LOSS_CAP:
            reasons.append(f"daily loss ${self._daily_loss:.2f} >= cap ${self.DAILY_LOSS_CAP}")

        return (len(reasons) == 0, reasons)

    def kelly_contracts(
        self,
        p_yes: float,
        yes_ask: float,
        no_ask: float,
        recommendation: str,
        bankroll: float,
    ) -> int:
        """Tiered Kelly sizing. Returns number of contracts (minimum 1 if Kelly is positive)."""
        if recommendation == "BUY_YES":
            cost  = yes_ask
            p_win = p_yes
        else:
            cost  = no_ask
            p_win = 1.0 - p_yes

        if cost <= 0 or cost >= 1:
            return 1

        b = (1.0 - cost) / cost   # net odds per unit staked
        q = 1.0 - p_win
        f = (p_win * b - q) / b   # full Kelly fraction

        if f <= 0:
            return 0

        # Tiered Kelly multiplier based on yes_ask zone
        if self.GOLDEN_ZONE_LOW <= yes_ask <= self.GOLDEN_ZONE_HIGH:
            tier = 0.35
        elif yes_ask <= 0.79:
            tier = 0.12
        elif yes_ask <= 0.85:
            tier = 0.08
        else:
            tier = 0.05

        dollar_bet      = f * tier * bankroll
        cost_per_ctr    = cost           # dollars per contract
        kelly_n         = dollar_bet / cost_per_ctr
        max_n           = (self.MAX_POSITION_PCT * bankroll) / cost_per_ctr

        return max(1, min(math.floor(kelly_n), math.floor(max_n)))
