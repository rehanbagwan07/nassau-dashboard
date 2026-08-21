import numpy as np

# Approximate centroid coordinates (lat, lon) for US states + Canadian provinces
# present in the dataset. Used to estimate customer-to-factory shipping distance
# since the raw data has no customer-level lat/long.
STATE_COORDS = {
    'Texas': (31.0, -100.0), 'Illinois': (40.0, -89.2), 'Pennsylvania': (41.2, -77.6),
    'Kentucky': (37.5, -85.3), 'Georgia': (32.9, -83.4), 'California': (36.8, -119.6),
    'Virginia': (37.5, -78.7), 'Delaware': (39.0, -75.5), 'South Carolina': (33.9, -80.9),
    'Ohio': (40.4, -82.9), 'Louisiana': (31.2, -92.0), 'Oregon': (44.0, -120.5),
    'Arizona': (34.2, -111.6), 'Arkansas': (34.9, -92.4), 'Michigan': (44.3, -85.6),
    'Tennessee': (35.9, -86.4), 'Florida': (27.8, -81.6), 'Ontario': (50.0, -85.0),
    'Indiana': (40.3, -86.1), 'Nevada': (39.3, -117.0), 'South Dakota': (44.3, -100.3),
    'New York': (42.9, -75.5), 'Wisconsin': (44.6, -89.9), 'Washington': (47.4, -120.5),
    'New Jersey': (40.1, -74.7), 'Missouri': (38.5, -92.5), 'North Carolina': (35.6, -79.4),
    'Colorado': (39.0, -105.5), 'Alberta': (55.0, -115.0), 'Utah': (39.3, -111.7),
    'Minnesota': (46.3, -94.3), 'Mississippi': (32.7, -89.6), 'Iowa': (42.0, -93.5),
    'New Mexico': (34.5, -106.1), 'Massachusetts': (42.3, -71.8), 'Alabama': (32.8, -86.8),
    'Idaho': (44.1, -114.6), 'Montana': (46.9, -110.0), 'Maryland': (39.0, -76.7),
    'Connecticut': (41.6, -72.7), 'New Hampshire': (43.9, -71.6), 'British Columbia': (54.0, -125.0),
    'Quebec': (52.9, -73.5), 'Nova Scotia': (45.0, -63.0), 'Oklahoma': (35.6, -97.5),
    'Nebraska': (41.5, -99.9), 'Maine': (45.4, -69.2), 'Kansas': (38.5, -98.4),
    'Rhode Island': (41.7, -71.5), 'Newfoundland and Labrador': (53.0, -60.0),
    'New Brunswick': (46.5, -66.2), 'Prince Edward Island': (46.5, -63.2),
    'District of Columbia': (38.9, -77.0), 'Vermont': (44.0, -72.7),
    'Manitoba': (55.0, -97.0), 'Saskatchewan': (54.0, -106.0), 'Wyoming': (43.0, -107.5),
    'North Dakota': (47.5, -100.5), 'West Virginia': (38.6, -80.6),
}

FACTORY_COORDS = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.11914, -96.18115),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.1175, -89.971107),
}

def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8  # miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def distance_to_factory(state, factory_name):
    if state not in STATE_COORDS or factory_name not in FACTORY_COORDS:
        return np.nan
    lat1, lon1 = STATE_COORDS[state]
    lat2, lon2 = FACTORY_COORDS[factory_name]
    return haversine(lat1, lon1, lat2, lon2)
