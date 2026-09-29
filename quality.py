import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

import create_plots as cp   # the group's note patterns, pipe specs, water properties and correlations

OUT = Path("quality_outputs")
PSI = cp.PSI_TO_PA          # Pa per psi
LPM = 1 / 60000             # m^3/s per L/min
INCH = 0.0254               # m per inch

# ---------------------------------------------------------------------------- instruments (datasheets)
# Side A = low-flow side: FTB2001 turbine meter and PX409 0-1 psid transducer.
# Side B = high-flow side: FV102 vortex meter and PX409 0-15 psid transducer.
DP_FS = {"A": 1.0, "B": 15.0}                          # transducer full scale, psid
UB_DP = {s: 0.0008 * DP_FS[s] * PSI for s in "AB"}      # 0.08% of full scale, Pa


def UB_Q(side, q):
    """Flowmeter U_B in L/min: turbine +-3% of reading; vortex +-2% of its 12 gpm full scale."""
    return 0.03 * q if side == "A" else 0.02 * 12 * 3.785411784


U_D = math.hypot(0.03e-3, 0.01e-3)                  # caliper 0.03 mm (+) repeatability of one reading, m
U_L = math.hypot(0.5 / 16 * INCH, 1 / 16 * INCH)    # half a 1/16-in mark (+) finding the tap centres, m
U_T_SPEC, U_T_POINT, U_T_FIXED = 0.03, 0.2, 0.3     # thermometer; T typed once per point; interpolated T (degC)
EPS = {"pvc": 4e-6 * INCH, "steel": 8e-6 * INCH, "copper": 8e-6 * INCH}   # group's comparator roughness, m
K_REFS = {"Perry's ½ open": 8.5, "Welty, wide open": 7.5, "NIBCO Cv, full open": 894 * 0.785**4 / 6.65**2}
DARBY = (1500.0, 1.70, 3.6)                          # Darby 3-K constants, globe valve, full open

# ---------------------------------------------------------------------------- planned runs (deck of Sep 29, slides 19-20)
PLAN_PIPES = {"PVC 0.470 in": ("pvc", 0.470, 66.0), "Steel 0.310 in": ("steel", 0.310, 79.5),
              "Copper 0.312 in": ("copper", 0.312, 72.5)}
PLAN_RUNS = {("PVC 0.470 in", "A"): [0.6, 0.9, 1.2, 1.4, 1.8, 2.0, 2.4, 3.6, 4.8],
             ("Steel 0.310 in", "A"): [0.6, 0.65, 0.7, 0.9, 1.1, 1.3, 1.6, 3.2, 4.8],
             ("Copper 0.312 in", "A"): [0.6, 0.65, 0.7, 0.9, 1.1, 1.3, 1.6, 3.2, 4.8],
             **{(p, "B"): [6.0, 11.0, 16.0, 21.0, 26.0, 30.0] for p in PLAN_PIPES}}
PLAN_VALVE_Q = [20.5, 22.5, 24.5, 26.5, 28.5, 30.5]    # same flows at every opening
PLAN_T, PLAN_UD = 23.0, 0.035e-3                       # degC; caliper U with Gehrke's rotation method, m
K_HALF, K_QUARTER = 7.0, 112.0                         # tour 1/2-open K; Perry's plug-disk 1/4-open K (worst case)

SAMPLE_POINTS = [("A", 0.285, 0.546), ("B", 0.41, 14.62)]    # (side, pipe label in the notes, ~L/min)
BUDGET_POINTS = [("A", 0.285, 0.546), ("A", 0.41, 0.875), ("A", 0.41, 2.93), ("B", 0.285, 6.61), ("B", 0.41, 20.65)]
LABEL = {0.41: "0.408 in", 0.285: "0.282 in"}
CORRS = ["Blasius", "Churchill", "Colebrook", "Haaland", "Romeo"]


# ============================================================================ math helpers
def kc(n):
    """Coverage factor for 95% confidence, same as Excel's T.INV.2T(0.05, n - 1)."""
    return stats.t.ppf(0.975, n - 1)


def course_round(x, u):
    """Course 3-30 rule: U keeps 1 or 2 digits so it reads 3-30 in its last place; x is rounded to match."""
    e = math.floor(math.log10(u))
    dec = -(e - 1) if round(u / 10 ** (e - 1)) <= 30 else -e
    d = max(dec, 0)
    return f"{round(x, dec):.{d}f} ± {round(u, dec):.{d}f}"


def colebrook(re, ed):
    """Colebrook, Fanning form: 1/sqrt(f) = -4 log10(eps/(3.7 D) + 1.255/(Re sqrt(f)))."""
    x = 4 * math.log10(re) - 0.4
    for _ in range(50):
        x = -4 * math.log10(ed / 3.7 + 1.255 * x / re)
    return 1 / x**2


def blasius(re):
    return 0.0791 * re**-0.25


def f_plan(re, ed):
    """Planning model: 16/Re laminar, Churchill in transition, Colebrook turbulent."""
    if re < 2100:
        return 16 / re
    return cp.churchill_fanning(re, ed) if re <= 4000 else colebrook(re, ed)


def darby(re):
    k1, ki, kd = DARBY
    return k1 / re + ki * (1 + kd / 0.75**0.3)       # 3/4-in valve


def fit_power(x, y):
    """Fit y = a x^n (LINEST on ln y vs ln x). Returns n, a, U_n = t(0.975, N-2) x standard error, residuals."""
    lx, ly = np.log(x), np.log(y)
    n, b = np.polyfit(lx, ly, 1)
    resid = ly - (n * lx + b)
    se = math.sqrt((resid**2).sum() / (len(lx) - 2) / ((lx - lx.mean())**2).sum())
    return n, math.exp(b), stats.t.ppf(0.975, len(lx) - 2) * se, resid


# ============================================================================ reading and cleaning the logs
def read_points(fname, side):
    """One point = the consecutive log lines that share a logging note."""
    pts = []
    for line in open(fname, encoding="utf-8").read().splitlines()[4:]:    # skip the 4 header lines
        t, q, p, note = [x.strip() for x in line.split("\t")]
        if not pts or note != pts[-1]["note"]:
            t_txt = cp.TEMP_PATTERN.search(note).group(1)
            valve = "globe" in note
            pts.append(dict(side=side, note=note, time=datetime.strptime(t, "%I:%M:%S %p"),
                            kind="valve" if valve else "pipe",
                            pipe="globe 1/2 open" if valve else float(cp.DIAMETER_PATTERN.search(note).group(1)),
                            T_note=float(t_txt), T_cut=t_txt.endswith("."), Q=[], P=[]))
        pts[-1]["Q"].append(float(q))
        pts[-1]["P"].append(float(p))
    return pts


def fix_temperatures(pts):
    """Both sides share one tank. A temperature typed unchanged for more than 10 min, or cut off ("23."),
    is replaced by the value interpolated in time from both sides' good notes."""
    for side in "AB":
        prev = None
        for p in (p for p in pts if p["side"] == side):
            if prev is None or p["T_note"] != prev["T_note"]:
                first = p["time"]
            p["stale"] = (p["time"] - first).total_seconds() > 600
            prev = p
    good = sorted((p["time"], p["T_note"]) for p in pts if not (p["stale"] or p["T_cut"]))
    t0 = good[0][0]
    tx, ty = [(t - t0).total_seconds() for t, _ in good], [T for _, T in good]
    for p in pts:
        fixed = p["stale"] or p["T_cut"]
        p["T"] = float(np.interp((p["time"] - t0).total_seconds(), tx, ty)) if fixed else p["T_note"]
        p["T_source"] = "interpolated" if fixed else "note"
        p["UA_T"] = U_T_FIXED if fixed else U_T_POINT
        p["U_T"] = math.hypot(U_T_SPEC, p["UA_T"])


# ============================================================================ one point: direct U, properties, propagation
def analyze(p):
    side = p["side"]
    Q, P = np.array(p["Q"]), np.array(p["P"])
    keep = Q >= 0.85 * np.median(Q)                  # FV102 dropouts: flow readings below 85% of the median
    Qk = Q[keep]
    r = dict(side=side, kind=p["kind"], pipe=p["pipe"], note=p["note"], time=p["time"].strftime("%H:%M:%S"),
             T_note=p["T_note"], T=p["T"], T_source=p["T_source"], UA_T=p["UA_T"], U_T=p["U_T"],
             n_Q=len(Qk), dropouts=int((~keep).sum()), Q_plain=Q.mean(),
             n_P=len(P), neg_dP=int((P < 0).sum()), dP_pos_only=P[P > 0].mean() * PSI)
    # direct measurements: U_A = k_c SSD/sqrt(n), U_B = datasheet, U_c = sqrt(U_A^2 + U_B^2)
    r.update(Q=Qk.mean(), SSD_Q=Qk.std(ddof=1), kc_Q=kc(len(Qk)))
    r["UA_Q"] = r["kc_Q"] * r["SSD_Q"] / math.sqrt(len(Qk))
    r["UB_Q"] = UB_Q(side, r["Q"])
    r["U_Q"] = math.hypot(r["UA_Q"], r["UB_Q"])
    r.update(dP=P.mean() * PSI, SSD_dP=P.std(ddof=1) * PSI, kc_dP=kc(len(P)))    # negative readings kept
    r["UA_dP"] = r["kc_dP"] * r["SSD_dP"] / math.sqrt(len(P))
    r["UB_dP"] = UB_DP[side]
    r["U_dP"] = math.hypot(r["UA_dP"], r["UB_dP"])
    r["valid"] = P.mean() < 0.98 * DP_FS[side]      # the tour's 1.3687-psi point sat past the 1-psid range
    # water properties at the point's temperature; their U from T +- U_T
    T, UT = r["T"], r["U_T"]
    r.update(rho=cp.density_kell(T), mu=cp.viscosity_kestin(T))
    r["U_rho"] = abs(cp.density_kell(T + UT) - cp.density_kell(T - UT)) / 2
    r["U_mu"] = abs(cp.viscosity_kestin(T + UT) - cp.viscosity_kestin(T - UT)) / 2
    # geometry, velocity, Re and F
    if p["kind"] == "pipe":
        r["D_in"], r["L_in"] = cp.PIPE_SPECS[p["pipe"]]["diameter_in"], cp.PIPE_SPECS[p["pipe"]]["length_in"]
    else:
        r["D_in"], r["L_in"] = cp.VALVE_DIAMETER_IN, float("nan")
    D = r["D_in"] * INCH
    fr = dict(Q=r["U_Q"] / r["Q"], dP=r["U_dP"] / abs(r["dP"]), D=U_D / D, rho=r["U_rho"] / r["rho"],
              mu=r["U_mu"] / r["mu"])
    r["v"] = r["Q"] * LPM / (math.pi * D**2 / 4)
    r["Re"] = r["rho"] * r["v"] * D / r["mu"]
    r["U_Re"] = r["Re"] * math.sqrt(fr["Q"]**2 + fr["D"]**2 + fr["rho"]**2 + fr["mu"]**2)
    r["F_hat"] = r["dP"] / r["rho"]
    r["U_F_hat"] = abs(r["F_hat"]) * math.hypot(fr["dP"], fr["rho"])
    r["Re_report"], r["F_hat_report"] = course_round(r["Re"], r["U_Re"]), course_round(r["F_hat"], r["U_F_hat"])
    if p["kind"] == "valve":                          # K = 2 dP/(rho v^2); (U_K/K)^2 = dP^2 + (4D)^2 + (2Q)^2 + rho^2
        r["K"] = 2 * r["dP"] / (r["rho"] * r["v"]**2)
        r.update(term_dP=fr["dP"], term_4D=4 * fr["D"], term_2Q=2 * fr["Q"], term_rho=fr["rho"])
        r["UK_rel"] = math.sqrt(r["term_dP"]**2 + r["term_4D"]**2 + r["term_2Q"]**2 + r["term_rho"]**2)
        r["U_K"] = r["K"] * r["UK_rel"]
        r["K_report"], r["K_darby"] = course_round(r["K"], r["U_K"]), darby(r["Re"])
        return r
    L = r["L_in"] * INCH                              # f = D dP/(2 L rho v^2), Fanning
    r["f"] = r["dP"] * D / (L * 2 * r["rho"] * r["v"]**2)
    terms = {"dP": fr["dP"], "5D": 5 * fr["D"], "2Q": 2 * fr["Q"], "L": U_L / L, "rho": fr["rho"]}
    r.update({f"term_{k}": v for k, v in terms.items()})
    r["Uf_rel"] = math.sqrt(sum(t**2 for t in terms.values()))
    r["U_f"] = r["f"] * r["Uf_rel"]
    r["largest"] = max(terms, key=terms.get)
    r["f_report"] = course_round(r["f"], r["U_f"])
    r["regime"] = "laminar" if r["Re"] < 2100 else "transition" if r["Re"] < 4000 else "turbulent"
    ed = EPS["pvc"] / D
    r.update(f_16Re=16 / r["Re"], f_Blasius=blasius(r["Re"]), f_Churchill=cp.churchill_fanning(r["Re"], ed),
             f_Colebrook=colebrook(r["Re"], ed), f_Haaland=cp.haaland_fanning(r["Re"], ed),
             f_Romeo=cp.romeo_royo_monzon_fanning(r["Re"], ed))
    for c in ["16Re"] + CORRS:
        r[f"inside_{c}"] = abs(r[f"f_{c}"] - r["f"]) <= r["U_f"]
    r["dP_HP"] = 128 * r["mu"] * L * r["Q"] * LPM / (math.pi * D**4)    # Hagen-Poiseuille
    return r


def closest(rows, side, pipe, q):
    return min((r for r in rows if r["side"] == side and r["pipe"] == pipe and r["valid"]), key=lambda r: abs(r["Q"] - q))


# ============================================================================ planned runs
def expected(pipe, side, q, noise, k_valve=None):
    """Expected dP and U_f/f (U_K/K for the valve) at a planned flow, at 23 C with the tour's noise levels."""
    rho, mu = cp.density_kell(PLAN_T), cp.viscosity_kestin(PLAN_T)
    U_dP = math.hypot(noise[side]["UA_dP"], UB_DP[side])
    U_Q = math.hypot(noise[side]["UA_Q_rel"] * q, UB_Q(side, q))
    if k_valve:
        D = cp.VALVE_DIAMETER_IN * INCH
        v = q * LPM / (math.pi * D**2 / 4)
        dP = k_valve * rho * v**2 / 2
        terms = [U_dP / dP, 2 * U_Q / q, 4 * PLAN_UD / D]
    else:
        mat, d_in, l_in = PLAN_PIPES[pipe]
        D, L = d_in * INCH, l_in * INCH
        v = q * LPM / (math.pi * D**2 / 4)
        dP = 2 * f_plan(rho * v * D / mu, EPS[mat] / D) * L * rho * v**2 / D
        terms = [U_dP / dP, 2 * U_Q / q, 5 * PLAN_UD / D, U_L / L]
    return dict(Re=rho * v * D / mu, dP_Pa=dP, dP_psi=dP / PSI, term_dP=terms[0], term_2Q=terms[1],
                term_D=terms[2], U_rel=math.sqrt(sum(t**2 for t in terms)))


def q_limit(pipe, side, noise):
    """Largest flow before the transducer's full scale (bisection)."""
    lo, hi = 0.05, 60.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if expected(pipe, side, mid, noise)["dP_psi"] > DP_FS[side] else (mid, hi)
    return lo


def valve_q_limit(k_valve):
    """Flow (L/min) at which a valve with loss coefficient K puts the 0-15 psid transducer at full scale."""
    rho, D = cp.density_kell(PLAN_T), cp.VALVE_DIAMETER_IN * INCH
    return math.sqrt(2 * DP_FS["B"] * PSI / (rho * k_valve)) * math.pi * D**2 / 4 / LPM


def plan_check(noise):
    out = []
    for (pipe, side), flows in PLAN_RUNS.items():
        for q in flows:
            e = expected(pipe, side, q, noise)
            flag = ("OVER transducer range" if e["dP_psi"] > DP_FS[side] else "U_f/f > 20%" if e["U_rel"] > 0.2 else "ok")
            out.append(dict(item=pipe, side=side, Q_LPM=q, **e, flag=flag))
    for label, k in (("Globe valve, tour 1/2-open K = 7", K_HALF), ("Globe valve 1/4 open if K = 112", K_QUARTER)):
        for q in PLAN_VALVE_Q:
            e = expected("valve", "B", q, noise, k_valve=k)
            flag = "OVER transducer range" if e["dP_psi"] > DP_FS["B"] else "U_K/K > 20%" if e["U_rel"] > 0.2 else "ok"
            out.append(dict(item=label, side="B", Q_LPM=q, **e, flag=flag))
    return out


# ============================================================================ written outputs
def instrument_table(valid):
    """Slide 1: U_B, U_A, U_c and fractional U for every direct measurement (medians over the tour points)."""
    out = []

    def add(qty, inst, unit, ub, ua, uc, val, frac):
        out.append(dict(quantity=qty, instrument=inst, unit=unit, U_B=np.median(ub), U_A=np.median(ua),
                        U_c=np.median(uc), typical=np.median(val), frac_U_median=np.median(frac),
                        frac_U_min=np.min(frac), frac_U_max=np.max(frac)))

    for side, dp_tag, q_tag in (("A", "PX409-001, 0-1 psid", "FTB2001 turbine"), ("B", "PX409-015, 0-15 psid", "FV102 vortex")):
        s = [r for r in valid if r["side"] == side]
        add(f"dP, side {side}", dp_tag, "Pa", [r["UB_dP"] for r in s], [r["UA_dP"] for r in s], [r["U_dP"] for r in s],
            [r["dP"] for r in s], [r["U_dP"] / abs(r["dP"]) for r in s])
        add(f"Q, side {side}", q_tag, "L/min", [r["UB_Q"] for r in s], [r["UA_Q"] for r in s], [r["U_Q"] for r in s],
            [r["Q"] for r in s], [r["U_Q"] / r["Q"] for r in s])
    add("T (fraction = effect on mu)", "Oakton thermometer", "°C", [U_T_SPEC], [r["UA_T"] for r in valid],
        [r["U_T"] for r in valid], [r["T"] for r in valid], [r["U_mu"] / r["mu"] for r in valid])
    for pipe in (0.41, 0.285):
        d_mm, l_mm = cp.PIPE_SPECS[pipe]["diameter_in"] * 25.4, cp.PIPE_SPECS[pipe]["length_in"] * 25.4
        add(f"D, {LABEL[pipe]}", "digital caliper", "mm", [0.03], [0.01], [U_D * 1e3], [d_mm], [U_D * 1e3 / d_mm])
        add(f"L, {LABEL[pipe]}", "tape, 1/16-in marks", "mm", [0.5 / 16 * 25.4], [1 / 16 * 25.4], [U_L * 1e3], [l_mm],
            [U_L * 1e3 / l_mm])
    return out


def pct(x):
    return f"{100 * x:.3f}%" if abs(x) < 1e-3 else f"{100 * x:.2f}%"


def sample_calc(r):
    """Slide 3: one pipe point written out step by step (symbols, numbers, rounded result)."""
    D, L = r["D_in"] * INCH, r["L_in"] * INCH
    fQ, fP, fD = r["U_Q"] / r["Q"], r["U_dP"] / abs(r["dP"]), U_D / D
    frho, fmu = r["U_rho"] / r["rho"], r["U_mu"] / r["mu"]
    ref = "16Re" if r["regime"] == "laminar" else "Churchill"
    ub_q = ("0.03 × Q (FTB2001: ±3% of reading)" if r["side"] == "A" else "0.02 × 12 gpm × 3.785 L/gal (FV102: ±2% of full scale)")
    head = f"Side {r['side']}, {LABEL[r['pipe']]} PVC: logged as \"{r['note']}\" ({r['regime']})"
    return "\n".join([
        head, "=" * len(head), "",
        "1) Flow rate:  U_A = k_c·SSD/√n,  k_c = T.INV.2T(0.05, n − 1);  U_c = √(U_A² + U_B²)",
        f"   Q = mean of {r['n_Q']} readings = {r['Q']:.4f} L/min;  SSD = {r['SSD_Q']:.4f};  k_c = {r['kc_Q']:.3f}",
        f"   U_A = {r['kc_Q']:.3f} × {r['SSD_Q']:.4f} / √{r['n_Q']} = {r['UA_Q']:.4f} L/min;   U_B = {ub_q} = {r['UB_Q']:.4f} L/min",
        f"   U_Q = √({r['UA_Q']:.4f}² + {r['UB_Q']:.4f}²) = {r['U_Q']:.4f} L/min   →   U_Q/Q = {pct(fQ)}", "",
        "2) Pressure drop",
        f"   ΔP = mean of {r['n_P']} readings × 6894.76 Pa/psi = {r['dP']:.1f} Pa;  SSD = {r['SSD_dP']:.1f} Pa;  k_c = {r['kc_dP']:.3f}",
        f"   U_A = {r['kc_dP']:.3f} × {r['SSD_dP']:.1f} / √{r['n_P']} = {r['UA_dP']:.1f} Pa;   "
        f"U_B = 0.0008 × {DP_FS[r['side']]:g} psid × 6894.76 = {r['UB_dP']:.1f} Pa (0.08% of full scale)",
        f"   U_ΔP = √({r['UA_dP']:.1f}² + {r['UB_dP']:.1f}²) = {r['U_dP']:.1f} Pa   →   U_ΔP/ΔP = {pct(fP)}", "",
        "3) Geometry and water properties",
        f"   D = {r['D_in']:.3f} in = {D * 1e3:.3f} mm ± {U_D * 1e3:.4f} mm (U_D/D = {pct(fD)});  "
        f"L = {r['L_in']:.1f} in = {L:.4f} m ± {U_L * 1e3:.2f} mm (U_L/L = {pct(U_L / L)})",
        f"   T = {r['T']:.2f} ± {r['U_T']:.2f} °C ({r['T_source']});  ρ (Kell) = {r['rho']:.2f} kg/m³ (±{pct(frho)});  "
        f"μ (Kestin) = {r['mu'] * 1e3:.4f} mPa·s (±{pct(fmu)})", "",
        "4) Velocity and Reynolds number",
        f"   v = 4Q/(πD²) = {r['v']:.4f} m/s;   Re = ρvD/μ = {r['rho']:.2f} × {r['v']:.4f} × {D:.5f} / {r['mu']:.4e} = {r['Re']:.0f}",
        f"   U_Re/Re = √((U_Q/Q)² + (U_D/D)² + (U_ρ/ρ)² + (U_μ/μ)²) = {pct(r['U_Re'] / r['Re'])}   →   Re = {r['Re_report']}", "",
        "5) Friction loss and Fanning friction factor",
        f"   F̂ = ΔP/ρ = {r['F_hat']:.4f} J/kg   →   F̂ = {r['F_hat_report']} J/kg",
        "   f = ¼(D/L)·ΔP/(ρv²/2)    [units: (m/m)·Pa/((kg/m³)(m/s)²) = dimensionless]",
        f"   f = 0.25 × ({D:.5f}/{L:.4f}) × {r['dP']:.1f} / ({r['rho']:.2f} × {r['v']:.4f}²/2) = {r['f']:.6f}",
        f"   U_f/f = √((U_ΔP/ΔP)² + (5U_D/D)² + (2U_Q/Q)² + (U_L/L)² + (U_ρ/ρ)²) = √({pct(r['term_dP'])}² + "
        f"{pct(r['term_5D'])}² + {pct(r['term_2Q'])}² + {pct(r['term_L'])}² + {pct(r['term_rho'])}²) = {pct(r['Uf_rel'])}",
        f"   Report (3-30 rule): f = {r['f_report']};   largest term: {r['largest']}",
        f"   Comparison: {ref} = {r['f_' + ref]:.6f} ({100 * (r['f'] / r['f_' + ref] - 1):+.1f}%); inside ±U_f: "
        f"{'yes' if r['inside_' + ref] else 'no'}", ""])


def write_csv(path, rows, keys):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:   # BOM so Excel shows ±, ° and ½
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ============================================================================ slide figures (course style, slide size)
def slide_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 20, "axes.labelsize": 21, "legend.fontsize": 17, "xtick.labelsize": 20, "ytick.labelsize": 20,
        "axes.linewidth": 1.3, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 7, "ytick.major.size": 7, "xtick.minor.size": 3.5, "ytick.minor.size": 3.5,
        "xtick.major.pad": 7, "ytick.major.pad": 6, "axes.grid": False, "legend.frameon": False, "savefig.dpi": 200,
        "mathtext.default": "regular", "legend.handlelength": 1.8, "legend.borderaxespad": 0.4,
        "legend.labelspacing": 0.35})


STYLE = {("A", 0.41): ("o", "white"), ("A", 0.285): ("s", "white"), ("B", 0.41): ("o", "black"), ("B", 0.285): ("s", "black")}
WHITE_BOX = dict(frameon=True, facecolor="white", edgecolor="none", framealpha=1.0)


def log_ticks(axis, ticks):
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    axis.set_minor_formatter(NullFormatter())


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)


def fig_dropouts(pts):
    """Slide 4: the worst FV102 dropout point."""
    Q = np.array(next(p for p in pts if p["side"] == "B" and "5.03" in p["note"])["Q"])
    keep = Q >= 0.85 * np.median(Q)
    t = np.arange(len(Q)) * 2.0
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    ax.axhline(Q[keep].mean(), color="black", ls="-", lw=1.4, label=f"Filtered mean, {Q[keep].mean():.2f}")
    ax.axhline(Q.mean(), color="0.4", ls=":", lw=2.4, label=f"Plain mean, {Q.mean():.2f}")
    ax.axhline(0.85 * np.median(Q), color="black", ls="--", lw=1.3, label="85% of median")
    ax.plot(t[keep], Q[keep], "o", mfc="black", mec="black", ms=8, label="Kept")
    ax.plot(t[~keep], Q[~keep], "o", mfc="white", mec="black", ms=9, mew=1.6, label="Rejected")
    ax.set(xlabel="Time into the logged point (s)", ylabel="Vortex-meter reading (L/min)", xlim=(-2, 68), ylim=(0, 10.5))
    ax.legend(loc="upper center", fontsize=16, ncol=2, columnspacing=1.0, handletextpad=0.4)
    save(fig, "s_dropouts.png")


def fig_budget(bud):
    """Slide 2: each term of U_f/f and the total, at five tour points."""
    fig, ax = plt.subplots(figsize=(6.3, 5.6))
    yb, h = np.arange(len(bud))[::-1] * 1.0, 0.24
    for y, r in zip(yb, bud):
        for j, (v, fc) in enumerate(zip([r["term_dP"], r["term_5D"], r["term_2Q"]], ["white", "0.6", "black"])):
            ax.barh(y + (1 - j) * h, 100 * v, height=h, color=fc, edgecolor="black", lw=1.0)
        tot = 100 * r["Uf_rel"]
        ax.plot([tot, tot], [y - 1.6 * h, y + 1.6 * h], color="black", lw=2.6)
        ax.text(tot + 0.6, y + 1.05 * h, f"{tot:.0f}%", fontsize=18, va="center", fontweight="bold")
    ax.set_yticks(yb)
    ax.set_yticklabels([f"{r['side']} · {LABEL[r['pipe']]}\n{r['Q']:.2g} L/min" for r in bud], fontsize=16)
    ax.tick_params(axis="y", length=0)
    ax.yaxis.set_ticks_position("left")
    ax.set(xlim=(0, 40), ylim=(-0.6, len(bud) - 0.25), xticks=[0, 10, 20, 30, 40], xlabel="Size of each term (%)")
    ax.legend(handles=[Patch(facecolor="white", edgecolor="black", label="ΔP term"),
                       Patch(facecolor="0.6", edgecolor="black", label="5·D term"),
                       Patch(facecolor="black", edgecolor="black", label="2·Q term"),
                       Line2D([], [], color="black", lw=2.6, label="U$_f$/f total")],
              loc="upper right", fontsize=16, handlelength=1.2, **WHITE_BOX)
    save(fig, "s_budget.png")


def fig_dp_vs_q(pipes, fits):
    """Slide 5: dP vs Q with error bars on both axes, Hagen-Poiseuille lines and side-B power-law fits."""
    fig, ax = plt.subplots(figsize=(7.9, 5.8))
    mu23 = cp.viscosity_kestin(23.0)
    for pipe in (0.41, 0.285):
        D, L = cp.PIPE_SPECS[pipe]["diameter_in"] * INCH, cp.PIPE_SPECS[pipe]["length_in"] * INCH
        q = np.array([0.45, 1.1 if pipe == 0.41 else 0.8])
        ax.plot(q, 128 * mu23 * L * q * LPM / (math.pi * D**4), "--", color="black", lw=1.5)
    for (side, pipe), (mk, fc) in STYLE.items():
        s = [r for r in pipes if r["side"] == side and r["pipe"] == pipe and r["valid"]]
        ax.errorbar([r["Q"] for r in s], [r["dP"] for r in s], xerr=[r["U_Q"] for r in s], yerr=[r["U_dP"] for r in s],
                    fmt=mk, mfc=fc, mec="black", ms=8.5, ecolor="black", elinewidth=1.1, capsize=3, lw=0,
                    label=f"{LABEL[pipe]}, side {side}")
    bad = [r for r in pipes if not r["valid"]]
    ax.plot([r["Q"] for r in bad], [r["dP"] for r in bad], "x", color="black", ms=12, mew=2.2, label="Saturated (dropped)")
    for pipe in (0.41, 0.285):
        n, a, U_n, qmin, qmax = fits[pipe]
        q = np.logspace(math.log10(qmin * 0.9), math.log10(qmax * 1.1), 50)
        ax.plot(q, a * q**n, "-", color="black", lw=1.5)
        if pipe == 0.285:
            ax.text(qmin * 0.78, a * qmin**n * 2.3, f"n = {n:.2f} ± {U_n:.2f}", fontsize=18, va="center", ha="right")
        else:
            ax.text(qmax * 1.18, a * (qmax * 1.18)**n * 0.95, f"n = {n:.2f} ± {U_n:.2f}", fontsize=18, va="center")
    ax.plot([], [], "--", color="black", lw=1.5, label="Hagen–Poiseuille")
    ax.plot([], [], "-", color="black", lw=1.5, label="Power-law fit, side B")
    ax.set(xscale="log", yscale="log", xlim=(0.4, 170), ylim=(20, 2e5),
           xlabel="Flow rate, Q (L/min)", ylabel="Pressure drop, ΔP (Pa)")
    log_ticks(ax.xaxis, [0.5, 1, 2, 5, 10, 20, 50, 100])
    h, lab = ax.get_legend_handles_labels()
    order = ["Hagen–Poiseuille", "Power-law fit, side B", "0.408 in, side A", "0.282 in, side A", "0.408 in, side B",
             "0.282 in, side B", "Saturated (dropped)"]
    ax.legend([h[lab.index(o)] for o in order], order, loc="lower right", fontsize=17)
    save(fig, "s_dP_vs_Q.png")


def fig_plan(plan_rows, limits, noise, pipe="Steel 0.310 in"):
    """Slide 7: expected U_f/f over each side's flow range for the planned steel pipe."""
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for side, (q0, q1) in (("A", (0.55, 5.0)), ("B", (4.5, 45.4))):
        q = np.logspace(math.log10(q0), math.log10(q1), 160)
        u = np.array([100 * expected(pipe, side, x, noise)["U_rel"] for x in q])
        cap = limits[pipe][side]
        ax.plot(q[q <= cap], u[q <= cap], "-", color="black", lw=2.6)
        ax.plot(q[q > cap], u[q > cap], ":", color="black", lw=2.0)
        ax.axvline(cap, color="black", lw=1.0)
        txt = f"{'1' if side == 'A' else '15'}-psid limit\n≈ {cap:.1f} L/min"
        if side == "A":
            ax.text(cap / 1.05, 43, txt, fontsize=16, va="top", ha="right")
        else:
            ax.text(cap * 1.05, 43, txt, fontsize=16, va="top", ha="left")
    for r in (r for r in plan_rows if r["item"] == pipe):
        over = r["flag"].startswith("OVER")
        ax.plot(r["Q_LPM"], 100 * r["U_rel"], "x" if over else "o", color="black", mfc="white",
                ms=11 if over else 9, mew=2.4 if over else 1.6, zorder=5)
    ax.text(0.6, 20.5, "Side A:\nturbine, 0–1 psid", fontsize=18, va="top")
    ax.text(7.0, 40.5, "Side B:\nvortex,\n0–15 psid", fontsize=18, va="top")
    ax.legend(handles=[Line2D([], [], color="black", lw=2.6, label="In range"),
                       Line2D([], [], color="black", lw=2.0, ls=":", label="Past the limit"),
                       Line2D([], [], color="black", marker="o", mfc="white", ms=9, lw=0, label="Planned"),
                       Line2D([], [], color="black", marker="x", ms=11, mew=2.4, lw=0, label="Planned, past limit")],
              loc="upper left", bbox_to_anchor=(0.0, 0.83), fontsize=16, **WHITE_BOX)
    ax.set(xscale="log", xlim=(0.5, 62), ylim=(0, 45), xlabel="Flow rate, Q (L/min)", ylabel="Expected U$_f$/f (%)")
    log_ticks(ax.xaxis, [0.5, 1, 2, 5, 10, 20, 50])
    save(fig, "s_plan_steel.png")


def fig_f_vs_re(pipes):
    """Slide 9: f vs Re with error bars on both axes, against 16/Re, Blasius and Churchill."""
    fig, ax = plt.subplots(figsize=(7.9, 5.8))
    re_l, re_t, re_a = (np.logspace(math.log10(a), math.log10(b), n) for a, b, n in ((900, 3500, 60), (3000, 7e4, 80), (900, 7e4, 150)))
    ax.plot(re_l / 1e3, 16e3 / re_l, "-", color="black", lw=1.8, label="16/Re")
    ax.plot(re_t / 1e3, 1e3 * blasius(re_t), "--", color="black", lw=1.8, label="Blasius")
    ax.plot(re_a / 1e3, [1e3 * cp.churchill_fanning(x, 1e-5) for x in re_a], ":", color="black", lw=2.4, label="Churchill")
    for (side, pipe), (mk, fc) in STYLE.items():
        s = [r for r in pipes if r["side"] == side and r["pipe"] == pipe and r["valid"]]
        ax.errorbar([r["Re"] / 1e3 for r in s], [1e3 * r["f"] for r in s], xerr=[r["U_Re"] / 1e3 for r in s],
                    yerr=[1e3 * r["U_f"] for r in s], fmt=mk, mfc=fc, mec="black", ms=8.5, ecolor="black",
                    elinewidth=1.1, capsize=3, lw=0, label=f"{LABEL[pipe]}, side {side}")
    ax.axvspan(2.1, 4.0, color="0.92", lw=0, zorder=0)
    ax.text(2.9, 4.25, "transition", fontsize=16, ha="center", va="bottom", color="0.25")
    ax.set(xscale="log", yscale="log", xlim=(0.9, 70), ylim=(4, 30), xlabel="Reynolds number, Re × 10$^{-3}$",
           ylabel="Fanning friction factor, f × 10$^{3}$")
    log_ticks(ax.xaxis, [1, 2, 5, 10, 20, 50])
    log_ticks(ax.yaxis, [4, 5, 6, 8, 10, 15, 20, 30])
    ax.legend(loc="upper right", fontsize=16, handletextpad=0.4, labelspacing=0.25)
    save(fig, "s_f_vs_Re.png")


def fig_roughness(noise, limits, pipe="Steel 0.310 in"):
    """Slide 10: can the planned data see roughness at D = 0.310 in? Returns each side's U_f/f range."""
    D = PLAN_PIPES[pipe][1] * INCH
    rho, mu = cp.density_kell(PLAN_T), cp.viscosity_kestin(PLAN_T)
    fig, ax = plt.subplots(figsize=(7.9, 5.8))
    bands = {}
    for side, q0 in (("A", 1.2), ("B", 6.0)):
        q = np.logspace(math.log10(q0), math.log10(limits[pipe][side]), 60)
        re = 4 * rho * q * LPM / (math.pi * D * mu)
        fs = np.array([colebrook(x, 1e-9) for x in re])
        u = np.array([expected(pipe, side, x, noise)["U_rel"] for x in q])
        ax.fill_between(re / 1e3, 1e3 * fs * (1 - u), 1e3 * fs * (1 + u), color="0.85", lw=0)
        ax.text(math.sqrt(re[0] * re[-1]) / 1e3, 3.35, f"side {side}: ±{100 * u.min():.0f}–{100 * u.max():.0f}%",
                fontsize=16, ha="center")
        bands[side] = (u.min(), u.max())
    re = np.logspace(math.log10(3000), math.log10(7e4), 120)
    for lab, eps, ls, lw in (("Smooth (PVC)", 1e-9, "-", 1.9), ("Drawn copper, 1.5 µm", 1.5e-6, "--", 1.7),
                             ("Stainless tubing, 2 µm", 2e-6, "-.", 1.7), ("Commercial steel, 45 µm", 45e-6, ":", 2.6)):
        ax.plot(re / 1e3, [1e3 * colebrook(x, eps / D) for x in re], ls, color="black", lw=lw, label=lab)
    ax.fill_between([], [], [], color="0.85", label="Our expected ±U$_f$")
    ax.set(xscale="log", yscale="log", xlim=(3, 70), ylim=(3, 22), xlabel="Reynolds number, Re × 10$^{-3}$",
           ylabel="Fanning friction factor, f × 10$^{3}$")
    log_ticks(ax.xaxis, [3, 5, 10, 20, 50])
    log_ticks(ax.yaxis, [4, 5, 6, 8, 10, 15, 20])
    ax.legend(loc="upper right", fontsize=16, labelspacing=0.25, **WHITE_BOX)
    save(fig, "s_roughness.png")
    return bands


def fig_valve(valves):
    """Backup slide: tour valve K with error bars against literature values."""
    fig, ax = plt.subplots(figsize=(7.9, 5.8))
    re = np.linspace(20, 38, 20)
    for (lab, k), ls in zip(K_REFS.items(), ["--", "-.", ":"]):
        ax.plot(re, [k] * len(re), ls, color="black", lw=1.9, label=f"{lab}, {k:.1f}")
    ax.plot(re, [darby(x * 1e3) for x in re], "-", color="black", lw=1.9, label="Darby 3-K, full open, 8.4")
    ax.errorbar([r["Re"] / 1e3 for r in valves], [r["K"] for r in valves], xerr=[r["U_Re"] / 1e3 for r in valves],
                yerr=[r["U_K"] for r in valves], fmt="o", mfc="white", mec="black", ms=10, ecolor="black",
                elinewidth=1.2, capsize=3, lw=0, label="Tour, \"½ open\"", zorder=5)
    ax.set(xlim=(19, 38), xticks=[20, 25, 30, 35], ylim=(5, 12), xlabel="Reynolds number in valve line, Re × 10$^{-3}$",
           ylabel="Loss coefficient, K")
    ax.legend(loc="upper left", fontsize=16, ncol=2, columnspacing=0.8, handletextpad=0.4, **WHITE_BOX)
    save(fig, "s_valve.png")


# ============================================================================ main: the numbers behind each slide
def main():
    OUT.mkdir(exist_ok=True)
    pts = read_points("FLU-prelab-Slowflow", "A") + read_points("FLU_Prelab_Highflow", "B")
    fix_temperatures(pts)
    rows = [analyze(p) for p in pts]
    pipes = [r for r in rows if r["kind"] == "pipe"]
    valid = [r for r in pipes if r["valid"]]
    valves = [r for r in rows if r["kind"] == "valve"]
    turb = [r for r in valid if r["regime"] == "turbulent"]
    slide_style()

    print("[Slide 1] Direct measurements (medians over valid tour points)")
    inst = instrument_table(valid + valves)
    for it in inst:
        print(f"  {it['quantity']:<28} U_B {it['U_B']:9.4g}  U_A {it['U_A']:9.4g}  U_c {it['U_c']:9.4g} {it['unit']:<6}"
              f" U_c/value {100 * it['frac_U_median']:.2f}% ({100 * it['frac_U_min']:.2f}-{100 * it['frac_U_max']:.2f}%)")
    ns = [r["n_Q"] for r in rows] + [r["n_P"] for r in rows]
    print(f"  readings per point: {min(ns)}-{max(ns)}; k_c = {kc(max(ns)):.3f}-{kc(min(ns)):.3f}")

    print("\n[Slide 2] Uncertainty budget (terms of U_f/f)")
    bud = [closest(rows, *bp) for bp in BUDGET_POINTS]
    for r in bud:
        print(f"  {r['side']} {LABEL[r['pipe']]} {r['Q']:6.2f} L/min  Re {r['Re']:6.0f}: dP {100 * r['term_dP']:4.1f}%  "
              f"5D {100 * r['term_5D']:3.1f}%  2Q {100 * r['term_2Q']:4.1f}%  ->  U_f/f {100 * r['Uf_rel']:4.1f}%")
    fig_budget(bud)

    print("\n[Slide 3] Sample calculations -> quality_outputs/quality_sample_calc.txt")
    samples = [closest(rows, *sp) for sp in SAMPLE_POINTS]
    for r in samples:
        print(f"  {r['side']} {LABEL[r['pipe']]} {r['Q']:.3f} L/min: f = {r['f_report']}, Re = {r['Re_report']}")

    print("\n[Slide 4] Data-quality filters")
    for r in rows:
        if r["dropouts"]:
            print(f"  {r['side']} {r['pipe']} Q={r['Q']:.2f}: {r['dropouts']} of {r['n_P']} flow readings are dropouts; "
                  f"plain mean {r['Q_plain']:.2f} L/min -> f {100 * ((r['Q'] / r['Q_plain'])**2 - 1):+.0f}% if not filtered")
        if not r["valid"]:
            print(f"  {r['side']} {r['pipe']} Q={r['Q']:.2f}: dP pinned at {r['dP'] / PSI:.4f} psi (past the "
                  f"{DP_FS[r['side']]:g}-psid range) -> dropped")
        if r["T_source"] == "interpolated":
            print(f"  {r['side']} {r['pipe']} Q={r['Q']:.2f}: T typed {r['T_note']:.2f} C (stale or cut off) -> {r['T']:.2f} C")
        if r["neg_dP"] > 5:
            print(f"  {r['side']} {r['pipe']} Q={r['Q']:.2f}: {r['neg_dP']} negative dP readings kept; mean {r['dP']:.1f} Pa "
                  f"(positives only would be {r['dP_pos_only']:.1f} Pa, {100 * (r['dP_pos_only'] / r['dP'] - 1):+.0f}%)")
    fig_dropouts(pts)

    print("\n[Slide 5] Power-law fits dP = a Q^n (U_n = t x standard error) and the laminar check")
    fits = {}
    for pipe in (0.41, 0.285):
        sb = [r for r in turb if r["pipe"] == pipe and r["side"] == "B"]
        n, a, U_n, _ = fit_power([r["Q"] for r in sb], [r["dP"] for r in sb])
        fits[pipe] = (n, a, U_n, min(r["Q"] for r in sb), max(r["Q"] for r in sb))
        both = [r for r in turb if r["pipe"] == pipe]
        n2, _, U_n2, _ = fit_power([r["Q"] for r in both], [r["dP"] for r in both])
        print(f"  {LABEL[pipe]} turbulent: side B n = {n:.3f} ± {U_n:.3f} (N={len(sb)});  "
              f"sides A+B pooled n = {n2:.3f} ± {U_n2:.3f} (N={len(both)})")
    s = math.log(colebrook(46000, 1e-9) / colebrook(12000, 1e-9)) / math.log(46000 / 12000)
    print(f"  smooth pipe over Re 12,000-46,000: f ~ Re^{s:.2f}, so dP ~ Q^{2 + s:.2f}")
    for r in (r for r in valid if r["regime"] == "laminar"):
        print(f"  laminar {r['side']} {LABEL[r['pipe']]} Q={r['Q']:.3f}: {r['dP']:.1f} ± {r['U_dP']:.1f} Pa vs "
              f"Hagen-Poiseuille {r['dP_HP']:.1f} Pa ({100 * (r['dP'] / r['dP_HP'] - 1):+.1f}%)")
    fig_dp_vs_q(pipes, fits)

    print("\n[Slide 6] Consistency checks")
    for pipe in (0.41, 0.285):
        sb = [r for r in turb if r["pipe"] == pipe and r["side"] == "B"]
        _, _, _, resid = fit_power([r["Re"] for r in sb], [r["f"] for r in sb])
        print(f"  {LABEL[pipe]} side B: scatter about a power law {100 * resid.std(ddof=2):.1f}% (1 SD); "
              f"U_f/f {100 * min(r['Uf_rel'] for r in sb):.0f}-{100 * max(r['Uf_rel'] for r in sb):.0f}%")
        for side in "AB":
            ratio = np.mean([r["f"] / r["f_Colebrook"] for r in turb if r["pipe"] == pipe and r["side"] == side])
            print(f"    side {side}: turbulent f / Colebrook = {ratio:.3f}")
    V = valves

    def slope(sel, y):
        return np.polyfit(np.log([r["Re"] for r in sel]), np.log(y), 1)[0]

    def k_slope(z):
        return slope(V, [r["K"] * (r["dP"] - z) / r["dP"] for r in V])

    def f_slope(z, pipe):
        sb = [r for r in pipes if r["side"] == "B" and r["pipe"] == pipe]
        return slope(sb, [r["f"] * (r["dP"] - z) / r["dP"] / r["f_Colebrook"] for r in sb])

    zs = np.linspace(-800, 200, 1001)
    z = zs[np.argmin([abs(k_slope(x)) for x in zs])]
    kz = [r["K"] * (r["dP"] - z) / r["dP"] for r in V]
    print(f"  valve K rises {100 * (V[-1]['K'] / V[0]['K'] - 1):.1f}% over Re {V[0]['Re']:.0f}-{V[-1]['Re']:.0f}; "
          f"f/f_Colebrook log-slopes (0 expected): 0.408 in {f_slope(0, 0.41):+.3f}, 0.282 in {f_slope(0, 0.285):+.3f}")
    print(f"  a zero offset z = {z:.0f} Pa ({z / PSI:+.3f} psi) makes K flat at {min(kz):.2f}-{max(kz):.2f} and moves the "
          f"f slopes to {f_slope(z, 0.41):+.3f} and {f_slope(z, 0.285):+.3f} (datasheet zero balance ±{0.005 * 15 * PSI:.0f} Pa)")

    print("\n[Slide 7] Planned runs (23 C, tour noise levels)")
    noise = {side: dict(UA_dP=float(np.median([r["UA_dP"] for r in rows if r["side"] == side and r["valid"]])),
                        UA_Q_rel=float(np.median([r["UA_Q"] / r["Q"] for r in rows if r["side"] == side and r["valid"]])))
             for side in "AB"}
    limits = {p: {s: q_limit(p, s, noise) for s in "AB"} for p in PLAN_PIPES}
    for p, lim in limits.items():
        print(f"  {p}: 1-psid limit (side A) {lim['A']:.1f} L/min; 15-psid limit (side B) "
              + (f"{lim['B']:.1f} L/min" if lim["B"] < 45 else "beyond the flowmeter's range"))
    plan = plan_check(noise)
    for r in plan:
        if r["flag"] != "ok":
            print(f"  {r['item']}, side {r['side']}, {r['Q_LPM']:g} L/min: {r['dP_psi']:.2f} psi, U {100 * r['U_rel']:.0f}% -> {r['flag']}")
    half = [100 * r["U_rel"] for r in plan if r["item"].startswith("Globe valve, tour")]
    print(f"  valve at {PLAN_VALVE_Q[0]}-{PLAN_VALVE_Q[-1]} L/min with K = 7: U_K/K {min(half):.1f}-{max(half):.1f}%; "
          f"15 psid is reached at {valve_q_limit(K_QUARTER):.1f} L/min if K = {K_QUARTER:.0f}")
    fig_plan(plan, limits, noise)

    print("\n[Slide 9] Correlations: share of points inside ±U_f and %AARD")
    summary = []
    for regime, names in (("laminar", ["16Re", "Churchill"]), ("transition", CORRS), ("turbulent", CORRS)):
        sel = [r for r in valid if r["regime"] == regime]
        for c in names:
            summary.append(dict(regime=regime, correlation=c, n=len(sel), inside=f"{sum(r['inside_' + c] for r in sel)}/{len(sel)}",
                                AARD_pct=100 * np.mean([abs(r["f_" + c] - r["f"]) / r["f"] for r in sel]),
                                mean_bias_pct=100 * np.mean([(r["f"] - r["f_" + c]) / r["f_" + c] for r in sel])))
            print(f"  {regime:10s} {c:10s} inside {summary[-1]['inside']:>5}   %AARD {summary[-1]['AARD_pct']:5.1f}")
    spread = [100 * (max(r["f_" + c] for c in CORRS) / min(r["f_" + c] for c in CORRS) - 1) for r in turb]
    print(f"  turbulent: correlations differ by {min(spread):.1f}-{max(spread):.1f}%; U_f/f "
          f"{100 * min(r['Uf_rel'] for r in turb):.0f}-{100 * max(r['Uf_rel'] for r in turb):.0f}% "
          f"(median {100 * np.median([r['Uf_rel'] for r in turb]):.0f}%)")
    for r in turb:
        if not r["inside_Churchill"]:
            print(f"  outside: side {r['side']} {LABEL[r['pipe']]} Re {r['Re']:.0f}: "
                  f"{100 * (r['f'] / r['f_Churchill'] - 1):+.1f}% vs Churchill, U_f/f {100 * r['Uf_rel']:.1f}%")
    fig_f_vs_re(pipes)

    print("\n[Slide 10] Roughness at D = 0.310 in (Colebrook, rise over a smooth pipe)")
    D = 0.310 * INCH
    for re in (4000, 50000):
        rise = [100 * (colebrook(re, e / D) / colebrook(re, 1e-9) - 1) for e in (1.5e-6, 2e-6, 45e-6)]
        print(f"  Re {re}: drawn copper {rise[0]:+.1f}%, stainless {rise[1]:+.1f}%, commercial steel {rise[2]:+.1f}%")
    bands = fig_roughness(noise, limits)
    print(f"  expected U_f/f for the planned steel runs: side A ±{100 * bands['A'][0]:.0f}-{100 * bands['A'][1]:.0f}%, "
          f"side B ±{100 * bands['B'][0]:.0f}-{100 * bands['B'][1]:.0f}%")

    print("\n[Backup] Globe valve, 1/2 open (raw K, includes pipe friction between the taps)")
    print(f"  K = {min(r['K'] for r in V):.2f}-{max(r['K'] for r in V):.2f} ± {min(r['U_K'] for r in V):.2f}-"
          f"{max(r['U_K'] for r in V):.2f} ({100 * min(r['UK_rel'] for r in V):.0f}-{100 * max(r['UK_rel'] for r in V):.0f}%) "
          f"at {V[0]['Q']:.1f}-{V[-1]['Q']:.1f} L/min")
    km = np.mean([r["K"] for r in V])
    for lab, k in list(K_REFS.items()) + [("Darby 3-K, full open", None)]:
        inside = sum(abs(r["K"] - (k or r["K_darby"])) <= r["U_K"] for r in V)
        ref = k or np.mean([r["K_darby"] for r in V])
        print(f"  mean K {km:.2f} vs {lab} {ref:.2f}: {100 * (km / ref - 1):+.1f}%, inside U_K at {inside}/{len(V)} points")
    fig_valve(V)

    # ---- tables (the figures were saved above, one per slide)
    point_keys = list(dict.fromkeys(k for r in rows for k in r))    # every field, in the order computed
    write_csv(OUT / "quality_point_table.csv", rows, point_keys)
    write_csv(OUT / "quality_instrument_table.csv", inst, list(inst[0]))
    write_csv(OUT / "quality_correlation_summary.csv", summary, list(summary[0]))
    write_csv(OUT / "quality_planned_run_check.csv", plan, list(plan[0]))
    (OUT / "quality_sample_calc.txt").write_text("FLU tour data: sample uncertainty calculations (quality.py)\n\n"
                                                 + "\n\n".join(sample_calc(r) for r in samples), encoding="utf-8")
    print(f"\nWrote tables and slide figures to {OUT.resolve()}")


if __name__ == "__main__":
    main()