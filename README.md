# Hajipoor et al. (2025) AHP–DEA replication / audit

This bundle contains a numerical reproduction of all aggregate numerical data
published in Tables 3–5 of the paper.

## Pipeline

1. Table 3 `RIM(normal)` is loaded.
2. Quality-attribute weights are recomputed as row averages.
3. Table 4 `CM` is loaded: **23 architecture styles × 7 attributes**.
4. Since the paper does not publish the individual expert RSM pairwise matrices,
   aggregate `RSM(total)` is reverse-reconstructed using Eq. (6):
   `RSM(total) = CM / W`.
5. Eq. (6) is checked by reconstructing `CM = W × RSM(total)`.
6. DEA inputs are `Cost` and `Development Effort`.
7. DEA outputs are `Security`, `Performance`, `Availability`,
   `Modifiability`, and `Usability`.
8. Eq. (9) is solved for every one of the 23 DMUs.
9. Recomputed scores are compared with the paper's Table 5.

## Important finding

The published Table 4 values plus the printed Eq. (9) do **not** independently
reproduce all published Table 5 scores under standard CCR input-oriented DEA.
The script leaves the discrepancy visible rather than adding undocumented
constraints.

## Run

```bash
pip install numpy pandas scipy
python hajipoor_ahp_dea_replication.py
```

All intermediate CSV files are written to `outputs/`.
