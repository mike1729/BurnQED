# miniF2F v2s Baseline Results

**Date:** 2026-02-28
**Model:** DeepSeek-Prover-V2-7B (base, no LoRA)
**Search:** Hybrid best-first, max_nodes=600, timeout=600s, 60 hybrid rounds
**Config:** `configs/search.toml` (hybrid_max_tokens=512, num_candidates=16, T=0.8)
**Benchmark:** `data/benchmarks/minif2f_v2s_test.json` (244 theorems)

## Summary

| Metric | Value |
|--------|-------|
| Proved | **148/244 (60.7%)** |
| Avg nodes (proved) | 2.9 |
| Median time (proved) | 21.2s |
| Mean time (proved) | 49.3s |
| Avg time (all) | 238.6s |
| 1-node solves | 110/148 (74%) |

## By Category

| Category | Proved | Total | Rate |
|----------|--------|-------|------|
| Algebra | 16 | 18 | 88.9% |
| MATHD Algebra | 53 | 70 | 75.7% |
| Induction | 6 | 8 | 75.0% |
| MATHD NT | 40 | 60 | 66.7% |
| NT | 5 | 8 | 62.5% |
| AMC12 | 19 | 45 | 42.2% |
| IMO | 7 | 20 | 35.0% |
| AIME | 2 | 15 | 13.3% |
| **Total** | **148** | **244** | **60.7%** |

## Failure Breakdown (96 theorems)

| Reason | Count | Notes |
|--------|-------|-------|
| timeout | 76 | Hit 600s wall clock |
| frontier_exhausted | 9 | All candidate tactics failed |
| max_rounds | 6 | Hit 60 hybrid rounds |
| error | 5 | Statement failed to parse in Lean |

### Errors (5)

Pantograph rejected the goal statement (0 nodes, 0 time):

- `aime_1997_p11`
- `amc12a_2008_p15`
- `mathd_algebra_144`
- `mathd_numbertheory_136`
- `mathd_numbertheory_530`

### Frontier Exhausted (9)

All candidate tactics were invalid; search had nowhere to go:

| Theorem | Nodes | Time |
|---------|-------|------|
| `mathd_numbertheory_232` | 1 | 4.0s |
| `mathd_numbertheory_33` | 1 | 1.8s |
| `mathd_numbertheory_668` | 1 | 5.5s |
| `induction_sum_odd` | 2 | 12.7s |
| `mathd_numbertheory_236` | 3 | 27.2s |
| `mathd_numbertheory_543` | 3 | 77.9s |
| `aime_1991_p6` | 4 | 109.5s |
| `mathd_numbertheory_198` | 5 | 60.7s |
| `induction_sum_1oktkp1` | 25 | 150.9s |

### Max Rounds (6)

Hit 60 hybrid rounds before timeout:

| Theorem | Nodes | Time |
|---------|-------|------|
| `imo_1987_p4` | 60 | 581.9s |
| `imo_1993_p5` | 60 | 416.4s |
| `mathd_algebra_11` | 60 | 556.9s |
| `mathd_algebra_214` | 60 | 475.2s |
| `mathd_algebra_480` | 60 | 421.1s |
| `mathd_algebra_51` | 60 | 300.5s |

## Failed Theorems

| Theorem | Nodes | Time | Reason |
|---------|-------|------|--------|
| `aimeII_2001_p3` | 22 | 625.9s | timeout |
| `aimeII_2020_p6` | 25 | 611.4s | timeout |
| `aimeI_2000_p7` | 51 | 600.2s | timeout |
| `aime_1983_p9` | 22 | 604.8s | timeout |
| `aime_1984_p5` | 16 | 600.2s | timeout |
| `aime_1987_p8` | 23 | 600.4s | timeout |
| `aime_1988_p3` | 32 | 627.6s | timeout |
| `aime_1988_p4` | 29 | 616.7s | timeout |
| `aime_1990_p2` | 36 | 604.6s | timeout |
| `aime_1991_p6` | 4 | 109.5s | frontier_exhausted |
| `aime_1994_p4` | 20 | 630.4s | timeout |
| `aime_1996_p5` | 18 | 613.8s | timeout |
| `aime_1997_p11` | 0 | 0.0s | error |
| `algebra_amgm_faxinrrp2msqrt2geq2mxm1div2x` | 36 | 600.3s | timeout |
| `algebra_amgm_sumasqdivbsqgeqsumbdiva` | 40 | 603.1s | timeout |
| `amc12_2000_p15` | 35 | 626.9s | timeout |
| `amc12_2001_p2` | 42 | 610.0s | timeout |
| `amc12a_2002_p12` | 22 | 629.4s | timeout |
| `amc12a_2002_p21` | 28 | 612.6s | timeout |
| `amc12a_2003_p24` | 27 | 614.9s | timeout |
| `amc12a_2003_p25` | 30 | 604.5s | timeout |
| `amc12a_2008_p15` | 0 | 0.0s | error |
| `amc12a_2008_p4` | 53 | 604.0s | timeout |
| `amc12a_2009_p15` | 17 | 604.8s | timeout |
| `amc12a_2009_p25` | 18 | 603.7s | timeout |
| `amc12a_2010_p10` | 36 | 601.1s | timeout |
| `amc12a_2010_p11` | 35 | 603.8s | timeout |
| `amc12a_2010_p22` | 19 | 607.9s | timeout |
| `amc12a_2011_p18` | 17 | 601.8s | timeout |
| `amc12a_2017_p7` | 41 | 601.2s | timeout |
| `amc12a_2019_p21` | 32 | 603.0s | timeout |
| `amc12a_2019_p9` | 40 | 620.9s | timeout |
| `amc12a_2020_p13` | 41 | 609.6s | timeout |
| `amc12a_2020_p21` | 54 | 602.8s | timeout |
| `amc12b_2002_p11` | 55 | 602.1s | timeout |
| `amc12b_2002_p3` | 14 | 645.3s | timeout |
| `amc12b_2002_p6` | 47 | 608.4s | timeout |
| `amc12b_2003_p17` | 57 | 606.3s | timeout |
| `amc12b_2004_p3` | 46 | 615.4s | timeout |
| `amc12b_2020_p5` | 56 | 609.4s | timeout |
| `amc12b_2021_p21` | 16 | 603.2s | timeout |
| `imo_1962_p4` | 19 | 611.1s | timeout |
| `imo_1964_p1_1` | 51 | 619.5s | timeout |
| `imo_1965_p1` | 20 | 656.1s | timeout |
| `imo_1973_p3` | 19 | 610.0s | timeout |
| `imo_1977_p5` | 15 | 640.9s | timeout |
| `imo_1978_p5` | 55 | 627.7s | timeout |
| `imo_1979_p1` | 55 | 605.6s | timeout |
| `imo_1987_p4` | 60 | 581.9s | max_rounds |
| `imo_1987_p6` | 24 | 624.9s | timeout |
| `imo_1988_p6` | 52 | 608.5s | timeout |
| `imo_1990_p3` | 18 | 616.9s | timeout |
| `imo_1993_p5` | 60 | 416.4s | max_rounds |
| `imo_2006_p3` | 22 | 611.5s | timeout |
| `induction_sum_1oktkp1` | 25 | 150.9s | frontier_exhausted |
| `induction_sum_odd` | 2 | 12.7s | frontier_exhausted |
| `mathd_algebra_11` | 60 | 556.9s | max_rounds |
| `mathd_algebra_144` | 0 | 0.0s | error |
| `mathd_algebra_149` | 43 | 612.3s | timeout |
| `mathd_algebra_151` | 30 | 618.3s | timeout |
| `mathd_algebra_214` | 60 | 475.2s | max_rounds |
| `mathd_algebra_282` | 19 | 604.4s | timeout |
| `mathd_algebra_31` | 19 | 605.2s | timeout |
| `mathd_algebra_323` | 39 | 608.2s | timeout |
| `mathd_algebra_327` | 32 | 605.2s | timeout |
| `mathd_algebra_393` | 53 | 602.2s | timeout |
| `mathd_algebra_421` | 27 | 626.4s | timeout |
| `mathd_algebra_422` | 40 | 609.5s | timeout |
| `mathd_algebra_437` | 31 | 601.1s | timeout |
| `mathd_algebra_480` | 60 | 421.1s | max_rounds |
| `mathd_algebra_509` | 20 | 609.6s | timeout |
| `mathd_algebra_51` | 60 | 300.5s | max_rounds |
| `mathd_algebra_59` | 52 | 612.1s | timeout |
| `mathd_numbertheory_126` | 20 | 605.7s | timeout |
| `mathd_numbertheory_13` | 20 | 620.4s | timeout |
| `mathd_numbertheory_136` | 0 | 0.0s | error |
| `mathd_numbertheory_156` | 22 | 623.9s | timeout |
| `mathd_numbertheory_198` | 5 | 60.7s | frontier_exhausted |
| `mathd_numbertheory_200` | 31 | 606.7s | timeout |
| `mathd_numbertheory_22` | 27 | 601.5s | timeout |
| `mathd_numbertheory_221` | 9 | 601.5s | timeout |
| `mathd_numbertheory_232` | 1 | 4.0s | frontier_exhausted |
| `mathd_numbertheory_236` | 3 | 27.2s | frontier_exhausted |
| `mathd_numbertheory_303` | 20 | 619.7s | timeout |
| `mathd_numbertheory_32` | 47 | 600.2s | timeout |
| `mathd_numbertheory_33` | 1 | 1.8s | frontier_exhausted |
| `mathd_numbertheory_405` | 14 | 631.8s | timeout |
| `mathd_numbertheory_48` | 27 | 609.1s | timeout |
| `mathd_numbertheory_530` | 0 | 0.0s | error |
| `mathd_numbertheory_543` | 3 | 77.9s | frontier_exhausted |
| `mathd_numbertheory_668` | 1 | 5.5s | frontier_exhausted |
| `mathd_numbertheory_709` | 45 | 611.8s | timeout |
| `mathd_numbertheory_780` | 48 | 607.7s | timeout |
| `numbertheory_aneqprodakp4_anmsqrtanp1eq2` | 58 | 607.6s | timeout |
| `numbertheory_sumkmulnckeqnmul2pownm1` | 23 | 614.4s | timeout |
| `numbertheory_xsqpysqintdenomeq` | 25 | 623.7s | timeout |

## Proved Theorems

| Theorem | Nodes | Time |
|---------|-------|------|
| `aime_1984_p15` | 1 | 83.2s |
| `aime_1991_p1` | 1 | 80.9s |
| `algebra_2complexrootspoly_xsqp49eqxp7itxpn7i` | 1 | 11.6s |
| `algebra_2rootsintpoly_am10tap11eqasqpam110` | 1 | 32.5s |
| `algebra_2rootspoly_apatapbeq2asqp2ab` | 1 | 38.3s |
| `algebra_2varlineareq_xpeeq7_2xpeeq3_eeq11_xeqn4` | 1 | 15.6s |
| `algebra_3rootspoly_amdtamctambeqnasqmbpctapcbtdpasqmbpctapcbta` | 1 | 18.9s |
| `algebra_amgm_prod1toneq1_sum1tongeqn` | 4 | 108.9s |
| `algebra_amgm_sqrtxymulxmyeqxpy_xpygeq4` | 4 | 52.9s |
| `algebra_apb4leq8ta4pb4` | 1 | 39.7s |
| `algebra_binomnegdiscrineq_10alt28asqp1` | 1 | 15.2s |
| `algebra_manipexpr_2erprsqpesqeqnrpnesq` | 1 | 15.2s |
| `algebra_manipexpr_apbeq2cceqiacpbceqm2` | 6 | 43.7s |
| `algebra_sqineq_2at2pclta2c2p41pc` | 1 | 18.5s |
| `algebra_sqineq_2unitcircatblt1` | 1 | 15.5s |
| `algebra_sqineq_36azm9asqle36zsq` | 1 | 15.3s |
| `algebra_sqineq_4bap1lt4bsqpap1sq` | 1 | 14.3s |
| `algebra_xmysqpymzsqpzmxsqeqxyz_xpypzp6dvdx3y3z3` | 1 | 22.8s |
| `amc12_2000_p11` | 2 | 26.4s |
| `amc12_2000_p5` | 1 | 11.4s |
| `amc12_2001_p9` | 3 | 20.3s |
| `amc12a_2002_p1` | 20 | 465.8s |
| `amc12a_2003_p1` | 46 | 239.4s |
| `amc12a_2008_p2` | 1 | 10.3s |
| `amc12a_2008_p8` | 13 | 221.9s |
| `amc12a_2009_p2` | 1 | 27.9s |
| `amc12a_2009_p5` | 1 | 29.1s |
| `amc12a_2009_p9` | 2 | 18.5s |
| `amc12a_2013_p7` | 1 | 14.3s |
| `amc12a_2013_p8` | 1 | 13.3s |
| `amc12a_2015_p10` | 1 | 19.9s |
| `amc12a_2016_p2` | 3 | 68.7s |
| `amc12a_2016_p3` | 1 | 20.5s |
| `amc12a_2017_p2` | 1 | 25.0s |
| `amc12a_2021_p7` | 1 | 41.3s |
| `amc12b_2003_p6` | 25 | 581.9s |
| `amc12b_2003_p9` | 1 | 9.6s |
| `imo_1961_p1` | 1 | 59.9s |
| `imo_1964_p1_2` | 2 | 35.0s |
| `imo_1966_p4` | 9 | 353.2s |
| `imo_1966_p5` | 7 | 271.0s |
| `imo_1967_p3` | 2 | 63.8s |
| `imo_1974_p5` | 2 | 91.9s |
| `imo_1984_p2` | 4 | 75.7s |
| `induction_divisibility_3div2tooddnp1` | 1 | 9.9s |
| `induction_divisibility_3divnto3m2n` | 1 | 10.1s |
| `induction_divisibility_9div10tonm1` | 1 | 11.1s |
| `induction_ineq_nsqlefactn` | 1 | 21.2s |
| `induction_seq_mul2pnp1` | 1 | 31.1s |
| `induction_sum2kp1npqsqm1` | 1 | 18.8s |
| `mathd_algebra_10` | 1 | 34.4s |
| `mathd_algebra_101` | 1 | 29.3s |
| `mathd_algebra_104` | 1 | 8.3s |
| `mathd_algebra_109` | 5 | 46.8s |
| `mathd_algebra_110` | 1 | 49.0s |
| `mathd_algebra_116` | 1 | 11.6s |
| `mathd_algebra_119` | 1 | 14.8s |
| `mathd_algebra_123` | 1 | 11.1s |
| `mathd_algebra_126` | 4 | 30.1s |
| `mathd_algebra_13` | 13 | 159.2s |
| `mathd_algebra_131` | 1 | 49.2s |
| `mathd_algebra_132` | 1 | 17.0s |
| `mathd_algebra_140` | 2 | 78.8s |
| `mathd_algebra_15` | 1 | 15.4s |
| `mathd_algebra_159` | 1 | 19.9s |
| `mathd_algebra_181` | 1 | 7.9s |
| `mathd_algebra_182` | 1 | 24.4s |
| `mathd_algebra_185` | 1 | 21.6s |
| `mathd_algebra_190` | 1 | 6.4s |
| `mathd_algebra_192` | 1 | 49.7s |
| `mathd_algebra_206` | 1 | 25.4s |
| `mathd_algebra_22` | 4 | 76.1s |
| `mathd_algebra_224` | 1 | 80.2s |
| `mathd_algebra_234` | 15 | 191.5s |
| `mathd_algebra_245` | 2 | 12.9s |
| `mathd_algebra_247` | 1 | 6.2s |
| `mathd_algebra_251` | 1 | 13.9s |
| `mathd_algebra_267` | 1 | 21.5s |
| `mathd_algebra_28` | 1 | 22.7s |
| `mathd_algebra_35` | 1 | 10.7s |
| `mathd_algebra_37` | 1 | 9.6s |
| `mathd_algebra_405` | 1 | 43.2s |
| `mathd_algebra_410` | 1 | 20.0s |
| `mathd_algebra_43` | 1 | 15.6s |
| `mathd_algebra_433` | 1 | 9.1s |
| `mathd_algebra_451` | 1 | 7.5s |
| `mathd_algebra_455` | 36 | 295.4s |
| `mathd_algebra_462` | 1 | 9.6s |
| `mathd_algebra_48` | 1 | 11.7s |
| `mathd_algebra_482` | 6 | 186.4s |
| `mathd_algebra_493` | 10 | 131.6s |
| `mathd_algebra_510` | 2 | 25.0s |
| `mathd_algebra_536` | 3 | 36.0s |
| `mathd_algebra_547` | 2 | 16.4s |
| `mathd_algebra_55` | 1 | 11.5s |
| `mathd_algebra_568` | 1 | 9.8s |
| `mathd_algebra_616` | 1 | 9.8s |
| `mathd_algebra_67` | 1 | 6.4s |
| `mathd_algebra_69` | 4 | 75.6s |
| `mathd_algebra_73` | 1 | 13.6s |
| `mathd_algebra_77` | 1 | 31.6s |
| `mathd_algebra_89` | 1 | 24.5s |
| `mathd_algebra_96` | 1 | 20.9s |
| `mathd_numbertheory_101` | 1 | 10.8s |
| `mathd_numbertheory_102` | 1 | 9.0s |
| `mathd_numbertheory_109` | 1 | 12.5s |
| `mathd_numbertheory_110` | 1 | 36.3s |
| `mathd_numbertheory_132` | 1 | 55.8s |
| `mathd_numbertheory_149` | 1 | 19.8s |
| `mathd_numbertheory_155` | 1 | 23.2s |
| `mathd_numbertheory_169` | 1 | 25.4s |
| `mathd_numbertheory_188` | 1 | 7.3s |
| `mathd_numbertheory_202` | 1 | 31.3s |
| `mathd_numbertheory_211` | 1 | 27.2s |
| `mathd_numbertheory_24` | 1 | 38.2s |
| `mathd_numbertheory_252` | 1 | 7.1s |
| `mathd_numbertheory_257` | 1 | 18.4s |
| `mathd_numbertheory_269` | 1 | 37.4s |
| `mathd_numbertheory_284` | 14 | 199.1s |
| `mathd_numbertheory_30` | 1 | 25.8s |
| `mathd_numbertheory_301` | 1 | 9.5s |
| `mathd_numbertheory_326` | 1 | 34.2s |
| `mathd_numbertheory_335` | 1 | 5.7s |
| `mathd_numbertheory_35` | 17 | 390.6s |
| `mathd_numbertheory_37` | 1 | 34.7s |
| `mathd_numbertheory_370` | 1 | 9.6s |
| `mathd_numbertheory_403` | 1 | 18.9s |
| `mathd_numbertheory_412` | 1 | 13.6s |
| `mathd_numbertheory_42` | 1 | 59.6s |
| `mathd_numbertheory_43` | 3 | 43.3s |
| `mathd_numbertheory_45` | 1 | 8.9s |
| `mathd_numbertheory_458` | 1 | 4.9s |
| `mathd_numbertheory_461` | 1 | 31.1s |
| `mathd_numbertheory_466` | 1 | 11.7s |
| `mathd_numbertheory_629` | 1 | 17.1s |
| `mathd_numbertheory_64` | 1 | 11.0s |
| `mathd_numbertheory_640` | 1 | 37.1s |
| `mathd_numbertheory_690` | 1 | 21.4s |
| `mathd_numbertheory_739` | 1 | 10.9s |
| `mathd_numbertheory_81` | 1 | 9.2s |
| `mathd_numbertheory_84` | 1 | 12.0s |
| `mathd_numbertheory_92` | 1 | 25.0s |
| `mathd_numbertheory_961` | 1 | 6.9s |
| `numbertheory_2dvd4expn` | 1 | 13.0s |
| `numbertheory_nckeqnm1ckpnm1ckm1` | 2 | 19.5s |
| `numbertheory_prmdvsneqnsqmodpeq0` | 2 | 10.0s |
| `numbertheory_sqmod3in01d` | 2 | 15.4s |
| `numbertheory_sqmod4in01d` | 12 | 100.7s |

