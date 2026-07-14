"""P7 unit test: the four-fold chi at the four pure amorphous/crystalline
limits, for BOTH bulk conventions, against the production evaluator.

Run:  python test_chi_limits.py   (or: pytest test_chi_limits.py)

Resolves the p1-vs-r14 convention question (P7 correction):
  * p1  -- chi_eff at the pure limits equals chi_aa / chi_ca / chi_ac /
           chi_cc directly: the four chi are ABSOLUTE pair interactions.
  * r14 -- the pure limits are chi_aa, chi_aa+chi_ca, chi_aa+chi_ac,
           chi_aa+chi_ca+chi_ac+chi_cc: the non-amorphous chi are
           DELTA-chi INCREMENTS relative to chi_aa.
"""
from coupled import chi_eff_limits

AA, AC, CA, CC = 1.2, 2.1, 2.6, 3.0
TOL = 1e-12


def test_p1_limits_are_absolute():
    lim = chi_eff_limits(AA, AC, CA, CC, bulk="p1")
    assert abs(lim["aa"] - AA) < TOL      # both amorphous
    assert abs(lim["ca"] - CA) < TOL      # i crystalline, j amorphous
    assert abs(lim["ac"] - AC) < TOL      # i amorphous, j crystalline
    assert abs(lim["cc"] - CC) < TOL      # both crystalline


def test_r14_limits_are_increments():
    lim = chi_eff_limits(AA, AC, CA, CC, bulk="r14")
    assert abs(lim["aa"] - AA) < TOL
    assert abs(lim["ca"] - (AA + CA)) < TOL
    assert abs(lim["ac"] - (AA + AC)) < TOL
    assert abs(lim["cc"] - (AA + CA + AC + CC)) < TOL


def main():
    for bulk in ("p1", "r14"):
        lim = chi_eff_limits(AA, AC, CA, CC, bulk=bulk)
        print(f"{bulk}: chi_eff pure limits = "
              f"aa={lim['aa']:.3f} ca={lim['ca']:.3f} "
              f"ac={lim['ac']:.3f} cc={lim['cc']:.3f}")
    test_p1_limits_are_absolute()
    test_r14_limits_are_increments()
    print("PASS: p1 limits are absolute; r14 limits are increments over chi_aa")


if __name__ == "__main__":
    main()
