"""Which facial region contains the whole-face hotspot? Uses the region maxima extracted by the FDA study."""
import pandas as pd

REGION_COLUMNS = {"inner canthi": "canthiMax", "forehead": "T_FH_Max", "mouth": "T_OR_Max"}


def hotspot_region_shares(df: pd.DataFrame, rnd: int = 1, tol: float = 0.05) -> dict:
    """A region 'contains' the hotspot when its own maximum equals the whole-face maximum (within tol, in C).
    Regions overlap, so the shares can add up to more than 1."""
    whole = df[f"T_Max{rnd}"]
    regs = pd.DataFrame({name: df[f"{col}{rnd}"] for name, col in REGION_COLUMNS.items()})
    ok = whole.notna() & regs.notna().all(axis=1)
    if not ok.any():
        raise ValueError("no complete rows")
    hit = regs[ok].sub(whole[ok], axis=0).abs().le(tol)
    return {"n": int(ok.sum()), "tolerance_c": tol,
            "matches_any": {k: float(hit[k].mean()) for k in REGION_COLUMNS},
            "none_of_the_three": float((~hit.any(axis=1)).mean())}
