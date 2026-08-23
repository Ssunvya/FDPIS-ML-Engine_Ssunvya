import argparse
import pandas as pd

from predict import FDPISPredictor


parser = argparse.ArgumentParser(
    description="FDPIS Flight Delay Prediction"
)

parser.add_argument("--day_of_month", type=int, default=15)
parser.add_argument("--day_of_week", type=int, default=4)
parser.add_argument("--month", type=int, default=8)

parser.add_argument("--dep_time", type=int, required=True)
parser.add_argument("--arr_time", type=int, required=True)

parser.add_argument("--distance", type=float, required=True)
parser.add_argument("--elapsed_time", type=float, required=True)

args = parser.parse_args()


def time_to_minutes(time_value):

    hour = time_value // 100
    minute = time_value % 100

    return hour * 60 + minute


dep_minutes = time_to_minutes(args.dep_time)
arr_minutes = time_to_minutes(args.arr_time)

dep_hour = args.dep_time // 100

weekend = 1 if args.day_of_week >= 6 else 0


flight = pd.DataFrame([{

    "DAY_OF_MONTH": args.day_of_month,
    "DAY_OF_WEEK": args.day_of_week,
    "MONTH": args.month,

    "DEP_MINUTES": dep_minutes,
    "ARR_MINUTES": arr_minutes,

    "DEP_HOUR_SIN":
        __import__("math").sin(
            2 * __import__("math").pi * dep_hour / 24
        ),

    "DEP_HOUR_COS":
        __import__("math").cos(
            2 * __import__("math").pi * dep_hour / 24
        ),

    "DOW_SIN":
        __import__("math").sin(
            2 * __import__("math").pi * args.day_of_week / 7
        ),

    "DOW_COS":
        __import__("math").cos(
            2 * __import__("math").pi * args.day_of_week / 7
        ),

    "IS_WEEKEND": weekend,

    "DISTANCE": args.distance,
    "CRS_ELAPSED_TIME": args.elapsed_time,

    "SCHEDULED_BLOCK_SPEED":
        args.distance / max(args.elapsed_time, 1)

}])


predictor = FDPISPredictor(
    model_dir="models"
)

probability = predictor.predict_proba(flight)[0]

prediction = predictor.predict(flight)[0]


print()
print("=" * 55)
print("     FDPIS FLIGHT DELAY PREDICTION RESULT")
print("=" * 55)

print(
    f"Scheduled Departure: {args.dep_time:04d}"
)

print(
    f"Scheduled Arrival:   {args.arr_time:04d}"
)

print(
    f"Distance:            {args.distance} miles"
)

print("-" * 55)

print(
    f"Delay Probability:   {probability * 100:.2f}%"
)

print(
    f"Decision Threshold:  {predictor.threshold:.2f}"
)

print()

if prediction == 1:

    print(
        "Prediction Status:  DELAYED (>= 15 minutes)"
    )

    print(
        "Risk Assessment:    HIGH DELAY RISK"
    )

else:

    print(
        "Prediction Status:  ON-TIME (< 15 minutes)"
    )

    print(
        "Risk Assessment:    LOW DELAY RISK"
    )

print("=" * 55)