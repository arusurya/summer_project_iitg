# Data Audit Report — 01_data_understanding

## File shapes
- loyalty  : (16737, 16)
- activity : (392936, 9)
- calendar : (2557, 4)

## Join key check
- Members in both tables        : 16,737
- Only in loyalty (no activity) : 0
- Only in activity (no profile) : 0

## Date range
- Activity covers: 2017 to 2018
- Dataset end month: 2018-12

## Behavioural churn preview
- Members inactive ≥6 months at dataset end: 899
- (Formal cancellation count will differ — see notebook 02)

## Issues to address in 02_data_cleaning
- [ ] Zero/negative CLV members
- [ ] Members active before enrollment date
- [ ] Duplicate grains in activity table (if any)
- [ ] Extreme outliers in flights/distance/points
- [ ] Confirm calendar table join logic