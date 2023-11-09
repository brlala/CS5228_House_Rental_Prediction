import numpy as np
import pandas as pd
import geopy.distance


def get_school_rank(df_school):
    # Set the Columns representing Registered/Vacancy Percentage in different phases
    percentage_cols = [
        '1_Registered_Vacancy_Percentage',
        '1_Registered_Vacancy_Percentage',
        '2B_Registered_Vacancy_Percentage',
        '2C_Registered_Vacancy_Percentage'
    ]

    # Assign weights to each phase: First phase get the greatest weight.
    weights = {
        '1_Registered_Vacancy_Percentage': 0.4,
        '2B_Registered_Vacancy_Percentage': 0.2,
        '2C_Registered_Vacancy_Percentage': 0.1,
    }

    # Calculate the weighted average Registered/Vacancy Percentage with given weights
    df_school['weighted_Percentage'] = sum(df_school[col] * wt for col, wt in weights.items())

    # Rank schools based on the weighted  average, with the highest average getting rank 1
    df_school['Rank'] = df_school['weighted_Percentage'].rank(ascending=False, method='min', na_option='bottom')

    # Display the ranked data
    school_rank = df_school[['name', 'weighted_Percentage', 'Rank']].sort_values(by='Rank')

    return school_rank


def haversine_vectorized(lat1, lon1, lat2s, lon2s):
    """
    Calculate the Haversine distance between one point and an array of points on the earth in km.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2s, lon2s = map(np.radians, [lat1, lon1, lat2s, lon2s])

    # Haversine formula
    dlat = lat2s - lat1
    dlon = lon2s - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2s) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Radius of Earth in kilometers. Use 3956 for miles
    r = 6371.0

    # Calculate the distance
    distances = r * c

    return distances


def calculate_distances_vectorized(house, df_schools):
    """
    Calculate distances between a house and all schools using vectorized operations.
    """
    latitudes = df_schools['latitude'].values
    longitudes = df_schools['longitude'].values

    distances = haversine_vectorized(house['latitude'], house['longitude'], latitudes, longitudes)

    return distances


def calculate_score_and_count_for_house(house, df_schools):
    # Calculate distances for all schools
    distances = calculate_distances_vectorized(house, df_schools)

    # Filter schools within 2 km
    schools_within_2km = df_schools[distances <= 2].copy()

    # Count the number of schools within 2km
    school_count = len(schools_within_2km)

    # If there are no schools within 2 km, return a score of 0
    if schools_within_2km.empty:
        return 0, school_count

    # Calculate location points for each school
    schools_within_2km['location_points'] = np.where(distances[distances <= 2] <= 1, 20, 10)

    # Get top 3 schools based on ranking
    top_2_schools = schools_within_2km.nsmallest(3, 'Ranking 2022')

    # Calculate total school and location points
    total_school_points = top_2_schools['school_points'].sum()
    total_location_points = top_2_schools['location_points'].sum()

    # Calculate final score
    final_score = 0.7 * total_school_points + 0.3 * total_location_points

    return final_score, school_count


def get_score_optimized(df_loc_primary_school, df_loc_house):
    # Calculate school ranking points
    n = len(df_loc_primary_school)
    df_loc_primary_school['school_points'] = n - df_loc_primary_school['Ranking 2022'] + 1

    # # Calculate scores for each house using the apply function
    # df_loc_house['score'] = df_loc_house.apply(calculate_score_for_house, axis=1, args=(df_loc_primary_school,))

    # Calculate scores and counts for each house
    df_loc_house[['school_score', 'school_count']] = df_loc_house.apply(lambda x: calculate_score_and_count_for_house(x, df_loc_primary_school), axis=1, result_type='expand')

    return df_loc_house
