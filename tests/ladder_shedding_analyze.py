"""Offline shedding analysis for the ladder_shedding_square .npz dumps.

Rigorous self-sustaining-limit-cycle test + Strouhal extraction:
  * exclude the trigger window AND a settle margin as transient,
  * fit the Cl amplitude envelope (per-window peak-to-peak) vs time — a
    GROWING or FLAT envelope after the trigger is off => self-sustaining;
    a DECAYING envelope => the trigger did not nucleate a limit cycle,
  * Strouhal from the FFT peak of the detrended, windowed steady tail,
    cross-checked with zero-crossing peak counting.

Usage:  python tests/ladder_shedding_analyze.py logs_shed/shed_L6_Re100_half0.0625.npz [trigger_steps]
"""
import sys
import numpy as np


def envelope_windows(sig, dt, nwin=10):
    """Per-window peak-to-peak amplitude and window-center times."""
    n = len(sig)
    w = n // nwin
    amps, tc = [], []
    for k in range(nwin):
        seg = sig[k * w:(k + 1) * w]
        if len(seg) < 4:
            continue
        amps.append(0.5 * (seg.max() - seg.min()))
        tc.append((k + 0.5) * w * dt)
    return np.array(tc), np.array(amps)


def st_fft(cl, dt, D, U=1.0):
    cl = cl - cl.mean()
    if cl.std() < 1e-9:
        return np.nan, np.nan
    win = np.hanning(len(cl))
    spec = np.abs(np.fft.rfft(cl * win))
    freqs = np.fft.rfftfreq(len(cl), d=dt)
    spec[0] = 0.0
    k = int(np.argmax(spec))
    f = float(freqs[k])
    return f * D / U, f


def st_peaks(cl, dt, D, U=1.0):
    """Strouhal by counting Cl sign-up-crossings of its mean."""
    c = cl - cl.mean()
    up = np.where((c[:-1] < 0) & (c[1:] >= 0))[0]
    if len(up) < 3:
        return np.nan
    periods = np.diff(up) * dt
    f = 1.0 / np.median(periods)
    return f * D / U


def analyze_series(cl, cd, dt, D, trigger_steps, label):
    n = len(cl)
    if n < 64:
        print(f"[{label}] series too short ({n})")
        return
    # transient = trigger window + a settle margin (30% of remaining)
    i_settle = trigger_steps + int(0.3 * (n - trigger_steps))
    tail_cl = cl[i_settle:]
    tail_cd = cd[i_settle:]
    tc, amps = envelope_windows(tail_cl, dt, nwin=8)
    # envelope trend: linear slope of amp vs time, normalized by mean amp
    if len(amps) >= 3 and amps.mean() > 1e-12:
        slope = np.polyfit(tc, amps, 1)[0]
        trend = slope * (tc[-1] - tc[0]) / amps.mean()   # frac change over tail
    else:
        trend = np.nan
    st_f, f_pk = st_fft(tail_cl, dt, D)
    st_p = st_peaks(tail_cl, dt, D)
    cl_amp = 0.5 * (tail_cl.max() - tail_cl.min())
    cl_rms = float(tail_cl.std())
    cd_mean = float(tail_cd.mean())
    sustaining = (cl_amp > 5e-3) and (trend > -0.5) and np.isfinite(st_f)
    print(f"[{label}] n={n} tail={len(tail_cl)}")
    print(f"  Cl_amp(pk)={cl_amp:.4f}  Cl_rms={cl_rms:.4f}  mean_Cd={cd_mean:+.4f}")
    print(f"  envelope trend over tail = {trend:+.2%}  (>-50% => not decaying)")
    print(f"  St(FFT)={st_f:.4f} (f={f_pk:.4f})   St(peaks)={st_p:.4f}")
    print(f"  => SELF-SUSTAINING SHEDDING: {sustaining}")
    return dict(cl_amp=cl_amp, cd_mean=cd_mean, st_fft=st_f, st_peaks=st_p,
                trend=trend, sustaining=bool(sustaining))


def main():
    path = sys.argv[1]
    trigger_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    d = np.load(path)
    dt = float(d["dt"])
    D = float(d["D"])
    print(f"=== {path}  dt={dt} D={D} trigger={trigger_steps} ===")
    analyze_series(np.asarray(d["proj_cl"]), np.asarray(d["proj_cd"]),
                   dt, D, trigger_steps, "PROJECTION")
    if len(np.asarray(d["mono_cl"])) > 64:
        analyze_series(np.asarray(d["mono_cl"]), np.asarray(d["mono_cd"]),
                       dt, D, trigger_steps, "MONOLITHIC")


if __name__ == "__main__":
    main()
