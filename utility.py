import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt

def load_flower_data(path: str):
	"""
	Load the CSV file with flower data
	columns=[   'Aanhef', 'Voorletters', 'Tussenvoegsel', 'Achternaam',
				'Straatnaam', 'Huisnummer', 'Toevoeging', 'Postcode', 'Woonplaats'  ]
	"""

	# Obtain the csv as a pandas dataframe
	df = pd.read_csv(path, delimiter=";")

	# Return the pandas dataframe
	return df


def enrich_flower_data(df: pd.DataFrame()):
	""" Enrich the flower dataframe with correct column types and by adding columns
	    - 'Name' with the full name of a person
	    - 'Address' with the full address of a person
	    - 'Size' with the number of flowers that need to be delivered """

	# Add Name column to the flower dataset
	df['Aanhef'] = df['Aanhef'].astype(str)
	df['Voorletters'] = df['Voorletters'].astype(str)
	df['Tussenvoegsel'] = df['Tussenvoegsel'].astype(str)
	df['Achternaam'] = df['Achternaam'].astype(str)
	df['Name'] = df[['Aanhef', 'Voorletters', 'Tussenvoegsel', 'Achternaam']].agg(' '.join, axis=1)
	df['Name'] = df['Name'].str.replace('nan ', '')

	# Add Address to the flower dataset
	df['Straatnaam'] = df['Straatnaam'].astype(str)
	df['Huisnummer'] = df['Huisnummer'].astype(str)
	df['Postcode'] = df['Postcode'].astype(str).str.replace(' ', '')
	df['Woonplaats'] = df['Woonplaats'].astype(str)
	df['Address'] = df[['Straatnaam', 'Huisnummer', 'Postcode', 'Woonplaats']].agg(' '.join, axis=1)
	df['Address'] = df['Address'].str.replace('nan ', '')

	# Add size of package
	df['Size'] = 1

	# Return enriched flower data
	return df


def add_geolocation_to_flower_dataframe(df: pd.DataFrame()):
	""" Transform the 'Address' column to geolocations, adding columns 'Longitude' and 'Latitude' to the dataframe """

	# Initialize geolocator object
	geolocator = Nominatim(user_agent="tulip_hackathon")

	# Initalize a dataframe with failed addresses
	fail_df = pd.DataFrame(columns=df.columns)

	# Initialize longitude and latitude columns
	df['Longitude'] = 0
	df['Latitude'] = 0

	# Loop over all addresses, and add the geolocations to the dataframe
	for index, row in df.iterrows():

		# Set location object
		location = geolocator.geocode(row["Address"])

		# Try to add longitude and latitude columns, otherwise add entire row to fail_df
		try:
			df.loc[index, 'Longitude'] = float(location.longitude)
			df.loc[index, 'Latitude'] = float(location.latitude)
		except AttributeError:
			fail_df = fail_df.append(row, ignore_index=True)

		# Print the index to keep track of the loop
		# print("Geo location conversions: " + str(index))

	# Return the dataframe with location columns and a dataframe with failed addresses
	return df, fail_df


def calculate_distances_flower_dataframe(df: pd.DataFrame()):
	""" Calculate the 'as the crow flies' distance matrix for all locations in the flower dataframe """

	# Initialize the distance matrix
	all_locations = np.arange(0, df.shape[0])
	distance_matrix = np.ones((df.shape[0], df.shape[0]))

	# Fill the distance matrix with bird's distances
	for i in all_locations:
		for j in all_locations:
			distance_matrix[i, j] = np.sqrt((df.loc[i, 'Longitude'] - df.loc[j, 'Longitude']) ** 2 +
			                                (df.loc[i, 'Latitude'] - df.loc[j, 'Latitude']) ** 2)

	# Return the distance matrix
	return distance_matrix, all_locations


def clarke_wright_cost_savings_heuristic(df: pd.DataFrame(), distance_matrix: np.array, all_locations: np.array):
	""" Calculate VRP solution by performing DHOO's implementation of the clark and wright cost savings heuristic """

	# Set number of locations
	N = len(all_locations)

	# Create dataframe with cost savings (columns "From", "To", "Savings")
	cost_savings = pd.DataFrame(columns=['From', 'To', 'Savings'], index=range(N ** 2))
	for i in all_locations:
		for j in all_locations:
			cost_savings.loc[i + (j * N), 'From'] = i
			cost_savings.loc[i + (j * N), 'To'] = j
			if i == j or i == 0 or j == 0:
				cost_savings.loc[i + (j*N), 'Savings'] = -1 * 10 ** 9
			else:
				cost_savings.loc[i + (j*N), 'Savings'] = distance_matrix[i, 0] + distance_matrix[0, j] - \
				                                         distance_matrix[i, j]

	# Sort dataframe with cost savings
	cost_savings = cost_savings.sort_values(by='Savings', ascending=False, na_position='last').reset_index(drop=True)
	print(cost_savings)

	# Add initial route to dataframe, which is going to every location from HQ and immediately returning to HQ
	max_capacity = 30
	df['coming_from'] = 0
	df['going_to'] = 0
	df['route_id'] = all_locations
	df['capacity_required'] = df['Size']

	# Keep adding savings from top to bottom (merging routes) if conditions allow it
	for i in range(N ** 2):

		# Obtain current location and next location for which routes need to be merged
		current_savings_row = cost_savings.loc[i, :]
		current_location = current_savings_row.From
		next_location = current_savings_row.To
		current_savings = current_savings_row.Savings

		# Check if conditions (capacity and start-end and same tour) for merging routes are not violated
		if df.loc[current_location, 'capacity_required'] + df.loc[next_location, 'capacity_required'] < max_capacity:
			if df.loc[current_location, 'going_to'] == 0 and df.loc[next_location, 'coming_from'] == 0:
				if df.loc[current_location, 'route_id'] != df.loc[next_location, 'route_id']:
					if current_savings > 0:

						# Update route_id
						route_id_1 = min(df.loc[current_location, 'route_id'], df.loc[next_location, 'route_id'])
						route_id_2 = max(df.loc[current_location, 'route_id'], df.loc[next_location, 'route_id'])
						df.loc[df['route_id'] == route_id_2, 'route_id'] = route_id_1

						# Update To and From locations
						df.loc[current_location, 'going_to'] = next_location
						df.loc[next_location, 'coming_from'] = current_location

						# Update capacity of route
						df.loc[df['route_id'] == route_id_1, 'capacity_required'] = \
							df.loc[current_location, 'capacity_required'] + df.loc[next_location, 'capacity_required']

	# Return dataframe with routes
	return df


def visualize_routes(df: pd.DataFrame()):
	""" Visualize the routes made with the clarke and wright heuristic """

	# Map route ID to color
	colors = ['b', 'g', 'c', 'm', 'y', 'b', 'g', 'c', 'm', 'y']
	route_ids = pd.unique(df['route_id'])
	color_dict = dict(zip(route_ids, colors))

	# Create a plot with routes
	plt.figure()
	for i in range(df.shape[0]):
		plt.plot(df.loc[i, 'Longitude'], df.loc[i, 'Latitude'], 'ro')
		j = df.loc[i, 'going_to']
		plt.plot([df.loc[i, 'Longitude'], df.loc[j, 'Longitude']], [df.loc[i, 'Latitude'], df.loc[j, 'Latitude']],
		         color_dict[df.loc[i, 'route_id']] + '-')
		k = df.loc[i, 'coming_from']
		plt.plot([df.loc[i, 'Longitude'], df.loc[k, 'Longitude']], [df.loc[i, 'Latitude'], df.loc[k, 'Latitude']],
		         color_dict[df.loc[i, 'route_id']] + '-')
	plt.plot(df.loc[0, 'Longitude'], df.loc[0, 'Latitude'], 'ko')
	plt.title("Baseline Routes Flowers Eindhoven")
	plt.xlabel("Longitude")
	plt.ylabel("Latitude")
	plt.show()


def get_vrp_solution_value(df: pd.DataFrame(), distance_matrix: np.array):
	""" Get the complete length of all the routes in the VRP solution """

	# Get a list of routes
	route_list = pd.unique(df['route_id'].tolist())

	# Calculate the length of each route in the VRP solution and add it to 'vrp_solution'
	vrp_solution = 0
	for route_id in route_list:
		if route_id > 0:

			# First location
			from_location = 0
			to_location = df[np.all([df['route_id'] == route_id,
			                         df['coming_from'] == from_location], axis=0)].index.tolist()[0]
			route_distance = distance_matrix[from_location, to_location]

			# As long as you are not back at HQ, keep adding distances of the route
			while to_location != 0:

				# Next location
				from_location = to_location
				to_location = df.loc[from_location, 'going_to']
				route_distance = route_distance + distance_matrix[from_location, to_location]

			# Add total route length to vrp_solution
			vrp_solution = vrp_solution + route_distance
			print("The VRP solution after route " + str(route_id) + " = " + str(vrp_solution))

	# Return the final solution value
	return vrp_solution
