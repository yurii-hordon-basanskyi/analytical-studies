import math


def calculate_distance(city_1, city_2):
    x1, y1 = map(math.radians, city_1)
    x2, y2 = map(math.radians, city_2)

    earth_radius = 6371.032

    calculated_distance = earth_radius * math.acos(math.sin(x1) * math.sin(x2) +
                                        math.cos(x1) * math.cos(x2) * math.cos(y1 - y2))

    return f"{calculated_distance:10.3f}"

kyiv_coords = (50.4546600, 30.5238000)
beijing_coords = (39.9075000, 116.3972300)
new_york_coords = (43.000000, -75.000000)
madrid_coords = (40.416775, -3.703790)
oslo_coords = (59.911491, 10.757933)

print(calculate_distance(kyiv_coords, beijing_coords))
print(calculate_distance(kyiv_coords, new_york_coords))
print(calculate_distance(kyiv_coords, madrid_coords))
print(calculate_distance(kyiv_coords, oslo_coords))
print(calculate_distance(oslo_coords, new_york_coords))
