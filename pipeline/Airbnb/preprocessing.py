from sklearn.preprocessing import LabelEncoder
import argparse
import logging
import os
import pandas as pd 

COLUMNS_PERCENT 	= 	["host_response_rate", "host_acceptance_rate"]
COLUMNS_CAT  		= 	[
							"host_is_superhost", "host_neighbourhood", "host_identity_verified",
							"host_response_time", "neighbourhood", "neighbourhood_cleansed",
							"property_type", "room_type", "bathrooms_text",
							"has_availability", "instant_bookable", "reviews_per_month"
						]
COLUMNS_NUMS 		= 	[
							"bedrooms", "beds", "review_scores_rating", "review_scores_accuracy", 
							"review_scores_cleanliness", "review_scores_checkin", "review_scores_communication",
							"review_scores_location", "review_scores_value", "host_listings_count", 
							"host_total_listings_count", "minimum_nights_avg_ntm", "maximum_nights_avg_ntm"
						]
COLUMNS_UNTOUCHED  	= 	[	
							"latitude", "longitude", "accommodates", 
							"availability_30", "availability_60", "availability_90", "availability_365", "number_of_reviews",
							"number_of_reviews_ltm", "number_of_reviews_l30d", "calculated_host_listings_count",
							"calculated_host_listings_count_entire_homes", "calculated_host_listings_count_private_rooms",
							"calculated_host_listings_count_shared_rooms", "price"
						]

COLUMNS 			= COLUMNS_PERCENT + COLUMNS_CAT + COLUMNS_NUMS + COLUMNS_UNTOUCHED

def percentage_encode(df, columns):
	fn = lambda col_name: lambda row: float(row[col_name].replace("%", "")) / 100

	for each_col in columns:
		df[each_col].fillna("0%", inplace = True)
		df[each_col] = df.apply(fn(each_col), axis = 1)
	return df

def mean_impute(df, columns):
	for each_col in columns:
		df[each_col].fillna(df[each_col].mean(), inplace = True)
	return df

def label_encode(df, columns):
	le = LabelEncoder()
	
	for each_col in columns:
		df[each_col] = le.fit_transform(df[each_col])
	return df

"""
python3 pipeline/Airbnb/preprocessing.py \
--input_csv /home/user/Desktop/tmp/Airbnb/Montreal.csv \
--output_dir /home/user/Desktop/sdb6/johan/Desktop/AdaptiveStream/data/airbnb \
--ratio 0.90
"""

if __name__ == "__main__":
	parser 	= argparse.ArgumentParser(description='Airbnb preprocessing pipeline')
	parser.add_argument('--input_csv', type = str, nargs = '?', help = 'Path to input .csv')
	parser.add_argument('--output_dir', type = str, nargs = '?', help = 'Path to output .csv')
	parser.add_argument('--ratio', type = float, nargs = '?', help = 'train : test split ratio')
	args 	= parser.parse_args()

	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
	logging.info(f"Using config: {args}")

	df 			= pd.read_csv(args.input_csv)
	df_filtered = df[COLUMNS]

	df_filtered = percentage_encode(df = df_filtered, columns = COLUMNS_PERCENT)
	df_filtered = label_encode(df = df_filtered, columns = COLUMNS_CAT)
	df_filtered = mean_impute(df = df_filtered, columns = COLUMNS_NUMS)
	df_filtered["price"] = df_filtered.apply(lambda row: float(row["price"].replace("$", "").replace(",", "")), axis = 1)

	df_train  		= df_filtered.sample(frac = args.ratio)
	df_test  		= df_filtered.drop(df_train.index)

	file_name 		= args.input_csv.split("/")[-1]
	output_train 	= os.path.join(args.output_dir, "train", file_name)
	output_test 	= os.path.join(args.output_dir, "test", file_name)

	df_train.to_csv(output_train, index = False)
	df_test.to_csv(output_test, index = False)