# Linear Regression Risk Scoring (No Training)

This project now uses a **linear regression style risk scorer** on top of existing detections.

Important: this is **not trained** with `fit()`.
- We use fixed coefficients (weights) chosen by domain logic.
- YOLO/detection pipeline remains unchanged.

## Why this was added

You asked for a classic ML-style algorithm (linear regression) without retraining the detector model.

So we added a separate numeric risk layer:
- Input: current frame detections + people count
- Output: `risk_score` in range 0 to 100 and `risk_level` (low/medium/high/critical)

## Where it is implemented

- Linear regression logic: `app/services/risk_regression.py`
- Runtime integration: `app/services/video_processor.py`
- API status fields: `app/schemas.py`, `app/routes/api.py`
- Dashboard metrics exposure: `app/services/analytics.py`, `app/routes/api.py`

## Core formula

For each frame, we compute:

`raw_risk = intercept + sum(weight_i * feature_i)`

Then:

`risk_score = clip(raw_risk, 0, 100)`

This is the exact linear regression form.

## Features used

The scorer builds numeric features from detections:

- `fire_count`
- `smoke_count`
- `flood_count`
- `road_accident_count`
- `earthquake_count`
- `crowd_panic_count`
- `fallen_person_count`
- `unsafe_zone_count`
- `survivor_count`
- `avg_disaster_conf` (average confidence excluding survivor class)
- `people_in_frame`

## Weights currently used

From `LinearRiskRegressor.COEFFICIENTS`:

- fire_count: 18.0
- smoke_count: 10.0
- flood_count: 16.0
- road_accident_count: 20.0
- earthquake_count: 18.0
- crowd_panic_count: 14.0
- fallen_person_count: 12.0
- unsafe_zone_count: 22.0
- survivor_count: 3.0
- avg_disaster_conf: 25.0
- people_in_frame: 1.0

Intercept:
- `INTERCEPT = 5.0`

## Risk level mapping

After score clipping:

- `score >= 80` -> `critical`
- `score >= 60` -> `high`
- `score >= 35` -> `medium`
- otherwise -> `low`

## Example

Suppose one frame has:

- 1 fire
- 1 unsafe zone
- 3 survivors
- people_in_frame = 3
- avg_disaster_conf = 0.82

Raw score:

`5 + (18*1) + (22*1) + (3*3) + (1*3) + (25*0.82)`

`= 5 + 18 + 22 + 9 + 3 + 20.5 = 77.5`

Final:

- `risk_score = 77.5`
- `risk_level = high`

## Why no training is required

Classical linear regression usually learns weights from data.

Here, we intentionally skip learning and set weights manually.
That means:

- No dataset preparation for this layer
- No training runtime
- Fully explainable behavior

Tradeoff:
- Fast and simple, but less statistically optimal than trained weights.

## API output impact

### `/api/status`
Now includes:
- `risk_score`
- `risk_level`

### `/api/dashboard`
`metrics` now includes:
- `risk_score`
- `risk_level`
- `risk_features`
- `risk_contributions`

`risk_contributions` shows each `(weight * feature)` term for explainability.

## Tuning without retraining

To adjust behavior, edit only these values in `app/services/risk_regression.py`:

- `INTERCEPT`
- `COEFFICIENTS`
- level thresholds in `_to_level`

No retraining command is needed.
