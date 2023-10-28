import numpy as np
import matplotlib.pyplot as plt

# Constants
EARTH_RADIUS = 6371  # in kilometers
MAX_LATITUDE = 90  # in degrees
MIN_LATITUDE = -90  # in degrees
MAX_ALTITUDE = 10  # in kilometers (considering troposphere)
SEASONS = ['Winter', 'Spring', 'Summer', 'Autumn']


def temperature_model(latitude, altitude, season):
    """
    A basic model to predict temperature based on latitude, altitude, and season.
    """
    # Base temperature based on latitude (equator is warmer, poles are colder)
    temp = 30 - (0.5 * abs(latitude))

    # Adjust temperature based on altitude (temperature decreases with altitude)
    temp -= 6.5 * altitude

    # Seasonal adjustments
    if season == 'Winter':
        temp -= 5
    elif season == 'Spring':
        temp += 2
    elif season == 'Summer':
        temp += 5
    elif season == 'Autumn':
        temp -= 2

    return temp


def simulate_weather_forecast():
    latitudes = np.linspace(MIN_LATITUDE, MAX_LATITUDE, 180)
    altitudes = np.linspace(0, MAX_ALTITUDE, 10)

    for season in SEASONS:
        temperatures = np.zeros((len(latitudes), len(altitudes)))

        for i, lat in enumerate(latitudes):
            for j, alt in enumerate(altitudes):
                temperatures[i, j] = temperature_model(lat, alt, season)

        plt.contourf(altitudes, latitudes, temperatures, 20, cmap='RdYlBu_r')
        plt.colorbar(label='Temperature (°C)')
        plt.xlabel('Altitude (km)')
        plt.ylabel('Latitude (°)')
        plt.title(f'Temperature Distribution - {season}')
        plt.show()


if __name__ == "__main__":
    simulate_weather_forecast()
