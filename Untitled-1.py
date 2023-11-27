# %%
from scipy.integrate import odeint
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import lombscargle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# %%
def simulate_system(initial_conditions, t):
    """
    Simulate the system of equations for a Newtonian potential and return the time series data.
    
    :param initial_conditions: List of initial conditions [x0, y0, v_x0, v_y0].
    :param t: Time points array for which to solve the equations of motion.
    :return: Tuple of time array and integrated values for x, y, v_x, and v_y.
    """
    def equations_of_motion(w, t):
        x, y, v_x, v_y = w
        r = (x**2 + y**2)**0.5
        return [v_x, v_y, -x/r**3, -y/r**3]

    # Integrate the equations of motion
    solution = odeint(equations_of_motion, initial_conditions, t)
    return t, solution[:, 0], solution[:, 1], solution[:, 2], solution[:, 3]


# %%
def compute_lomb_scargle(t, x, y, freq_count=1000):
    """
    Compute the Lomb-Scargle periodogram for the provided time series data.

    :param t: Time points array.
    :param x: X position data array.
    :param y: Y position data array.
    :param freq_count: Number of frequency bins to use for the periodogram.
    :return: Frequencies and the power spectrum (periodogram) of the signal.
    """
    # We will consider the system's kinetic energy time series as the signal.
    # Kinetic energy, KE = 0.5 * m * (v_x^2 + v_y^2), assuming mass (m) = 1 for simplicity.
    v_x = np.gradient(x, t)
    v_y = np.gradient(y, t)
    ke = 0.5 * (v_x**2 + v_y**2)
    
    # Normalize the signal
    ke_normalized = ke - np.mean(ke)

    # Define the frequency domain
    freqs = np.linspace(0.01, 10, freq_count)
    
    # Compute Lomb-Scargle periodogram
    power = lombscargle(t, ke_normalized, freqs, normalize=True)
    
    return freqs, power

# %%
def find_peak_frequency(freqs, power):
    """
    Find the peak frequency from the Lomb-Scargle periodogram.

    :param freqs: Frequencies array from the periodogram.
    :param power: Power spectrum from the periodogram.
    :return: Peak frequency.
    """
    # Find the index of the maximum power
    peak_index = np.argmax(power)
    # Return the corresponding frequency
    return freqs[peak_index]


# %%
def extract_features(x, y, v_x, v_y):
    """
    Extract features from the time series data for use in machine learning models.

    :param x: X position data array.
    :param y: Y position data array.
    :param v_x: X velocity data array.
    :param v_y: Y velocity data array.
    :return: Extracted features.
    """
    # Here we use simple statistical features, but you could add more complex ones if needed.
    features = {
        'x_mean': np.mean(x),
        'y_mean': np.mean(y),
        'x_var': np.var(x),
        'y_var': np.var(y),
        'x_skew': skew(x),
        'y_skew': skew(y),
        'x_kurt': kurtosis(x),
        'y_kurt': kurtosis(y),
        'v_x_mean': np.mean(v_x),
        'v_y_mean': np.mean(v_y),
        'v_x_var': np.var(v_x),
        'v_y_var': np.var(v_y),
        'v_x_skew': skew(v_x),
        'v_y_skew': skew(v_y),
        'v_x_kurt': kurtosis(v_x),
        'v_y_kurt': kurtosis(v_y),
    }
    
    return features


# %%
def subsample_time_series(t, x, y, v_x, v_y, subsample_size=30):
    """
    Subsample the time series data to simulate incomplete observations.

    :param t: Time points array.
    :param x: X position data array.
    :param y: Y position data array.
    :param v_x: X velocity data array.
    :param v_y: Y velocity data array.
    :param subsample_size: The number of data points to subsample to.
    :return: Subsampled time series data.
    """
    # Ensure we always subsample the same way for reproducibility
    np.random.seed(0)
    
    # Choose random indices for subsampling
    subsample_indices = np.random.choice(len(t), size=subsample_size, replace=False)
    subsample_indices.sort()  # Sort the indices to maintain the time order
    
    return t[subsample_indices], x[subsample_indices], y[subsample_indices], v_x[subsample_indices], v_y[subsample_indices]

# %%
# Number of samples to generate
num_samples = 100000
# Number of observations in each subsample
subsample_size = 30

# Prepare lists to hold the data
data_features = []
data_peak_freqs = []

# Generate the dataset
for i in range(num_samples):
    # Generate random initial conditions within a reasonable range
    np.random.seed(i)
    initial_conditions = np.random.rand(4)  
    # Simulate the system
    t_max = np.random.uniform(5, 20)  # Random end time for the observations
    t = np.linspace(0, t_max, 1000)  # Assume we have 1000 potential observations
    t, x, y, v_x, v_y = simulate_system(initial_conditions, t)
    
    # Subsample the time series data to simulate incomplete observations
    t_sub, x_sub, y_sub, v_x_sub, v_y_sub = subsample_time_series(t, x, y, v_x, v_y, subsample_size)
    
    # Compute the Lomb-Scargle periodogram on the subsampled data
    freqs, power = compute_lomb_scargle(t_sub, x_sub, y_sub)
    
    # Find the peak frequency
    peak_freq = find_peak_frequency(freqs, power)
    
    # Extract features from the subsampled time series data
    features = extract_features(x_sub, y_sub, v_x_sub, v_y_sub)
    
    # Store the features and peak frequency
    data_features.append(features)
    data_peak_freqs.append(peak_freq)

# Convert lists to a DataFrame
features_df = pd.DataFrame(data_features)
features_df['PeakFrequency'] = data_peak_freqs

# # Save the DataFrame to a CSV file
# features_df.to_csv('simulated_time_series_data.csv', index=False)


# %%
features_df

# %%
features_df['PeakFrequency'].describe()

# %%
plt.hist(features_df['PeakFrequency'], bins=100)

# %%
def plot_solution(initial_conditions, t_max, num_points=100):
    """
    Plots the solution of the Newtonian potential system for given initial conditions.

    :param initial_conditions: List of initial conditions [x0, y0, v_x0, v_y0].
    :param t_max: Maximum time for the simulation.
    :param num_points: Number of points to generate in the time array.
    """
    # Generate time array
    t = np.linspace(0, t_max, num_points)

    # Use the simulate_system function to get the solution
    t, x, y, v_x, v_y = simulate_system(initial_conditions, t)

    # Plotting
    fig, axes = plt.subplots(5, 1, figsize=(5, 17))

    axes[0].plot(x, y, 'o', ms=1)
    axes[1].plot(t, x)
    axes[2].plot(t, y)
    axes[3].plot(t, v_x)
    axes[4].plot(t, v_y)

    axes[0].set_ylabel('y', fontsize=16)
    axes[0].set_xlabel('x', fontsize=16)
    axes[1].set_ylabel('x', fontsize=16)
    axes[2].set_ylabel('y', fontsize=16)
    axes[3].set_ylabel('v_x', fontsize=16)
    axes[4].set_ylabel('v_y', fontsize=16)

    for ax in axes[1:]:
        ax.set_xlabel('t', fontsize=16)

    plt.tight_layout()
    plt.show()


# %%
initial_conditions = np.random.rand(4) 
t_max = 10 
plot_solution(initial_conditions, t_max)

# %%
data = features_df.copy()

# %%
def remove_outliers(df, column_names):
    """
    Remove outliers from a pandas DataFrame based on the interquartile range.

    :param df: Pandas DataFrame.
    :param column_names: List of column names to check for outliers.
    :return: DataFrame with outliers removed.
    """
    clean_df = df.copy()
    
    for column in column_names:
        Q1 = clean_df[column].quantile(0.25)
        Q3 = clean_df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter out the outliers
        clean_df = clean_df[(clean_df[column] >= lower_bound) & (clean_df[column] <= upper_bound)]
    
    return clean_df

data = features_df.copy()
# %%
data = remove_outliers(data, ['PeakFrequency'])

# %%
data['PeakFrequency'].describe()

# %%
data.dropna(inplace=True)

# %%
# Split the data into features and target
X = data.drop('PeakFrequency', axis=1)
y = data['PeakFrequency']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
# MAPE
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f"Mean Absolute Percentage Error: {mape}")
# R2
r2 = rf_regressor.score(X_test, y_test)
print(f"R2 Score: {r2}")

# %%
# Plot histogram of the test and of the predicted values
plt.hist(y_test, bins=100, label='test')    
plt.hist(y_pred, bins=100, label='predicted')

# %%
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) 
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions')
plt.show()

# %%
# Set the initial conditions
initial_conditions = [1.0, 0.0, 0.0, 0.1]
t_max = 10  # value that we used for training
num_points = 100000  # same we used in training
subsample_size = 30  # same we used in training

# Generate the time series data
t, x, y, v_x, v_y = simulate_system(initial_conditions, np.linspace(0, t_max, num_points))

# Subsample the data (if your model was trained on subsampled data)
t_sub, x_sub, y_sub, v_x_sub, v_y_sub = subsample_time_series(t, x, y, v_x, v_y, subsample_size)

# Extract features
features = extract_features(x_sub, y_sub, v_x_sub, v_y_sub)

# Convert features to a format suitable for the model (e.g., DataFrame)
features_df = pd.DataFrame([features])

# Predict with the model
predicted_peak_frequency = rf_regressor.predict(features_df)

print(f"Predicted Peak Frequency: {predicted_peak_frequency[0]}")


# %%
def simulate_system(func, initial_conditions, t, params):
    """
    Generalized simulation function for different potentials.

    :param func: Function representing the equations of motion.
    :param initial_conditions: Initial conditions for the simulation.
    :param t: Array of time points for the simulation.
    :param params: Parameters required for the specific potential.
    :return: Simulated time series data.
    """
    solution = odeint(func, initial_conditions, t, args=params)
    return t, solution[:,0], solution[:,1], solution[:,2], solution[:,3]


# %%
def simulate_harmonic_oscillator(initial_conditions, t, omega, a):
    def equations_of_motion(xyv_xv_y, t, omega, a):
        x, y, v_x, v_y = xyv_xv_y
        return [v_x, v_y, a - omega**2 * x, a - omega**2 * y]
    
    return simulate_system(equations_of_motion, initial_conditions, t, (omega, a))


# %%
def simulate_modified_newtonian(initial_conditions, t, b, a):
    def equations_of_motion(xyv_xv_y, t, b, a):
        x, y, v_x, v_y = xyv_xv_y
        return [v_x, v_y, -x / (a * b**2 + x**2 + y**2)**(3/2), -y / (b**2 + x**2 + y**2)**(3/2)]

    return simulate_system(equations_of_motion, initial_conditions, t, (b, a))

# %%
def simulate_custom_potential(initial_conditions, t, c, r0, alpha):
    def equations_of_motion(xyv_xv_y, t, c, r0, alpha):
        x, y, v_x, v_y = xyv_xv_y
        r = np.sqrt(x**2 + y**2)
        return [v_x, v_y, -(c * alpha * x) * (r0 / r)**(alpha - 1) * r0 / r**3,
                -(c * alpha * y) * (r0 / r)**(alpha - 1) * r0 / r**3]

    return simulate_system(equations_of_motion, initial_conditions, t, (c, r0, alpha))


# %%
def simulate_disk_galaxy(initial_conditions, t, vo, Rc, q):
    def equations_of_motion(rzv_rv_z, t, vo, Rc, q):
        r, z, v_r, v_z = rzv_rv_z
        return [v_r, v_z, (-vo**2 * r) / (Rc**2 + r**2 + (z/q)**2),
                (-vo**2 * z) / (q**2 * (Rc**2 + r**2 + (z/q)**2))]

    return simulate_system(equations_of_motion, initial_conditions, t, (vo, Rc, q))


# %%
def simulate_another_disk_galaxy(initial_conditions, t, vo, Rc, q):
    def equations_of_motion(rzv_rv_z, t, vo, Rc, q):
        r, z, v_r, v_z = rzv_rv_z
        return [v_r, v_z,
                -vo**2 * (-2 * r * np.sqrt(r**2 + z**2) - ((r * (r**2 - z**2)) / np.sqrt(r**2 + z**2)) + 2 * r) /
                (2 * (Rc**2 + (z/q)**2 - ((r**2 - z**2) * np.sqrt(r**2 + z**2)) + r**2)),
                -vo**2 * (2 * z * np.sqrt(r**2 + z**2) - ((z * (r**2 - z**2)) / np.sqrt(r**2 + z**2)) + 2 * z / q**2) /
                (2 * (Rc**2 + (z/q)**2 - ((r**2 - z**2) * np.sqrt(r**2 + z**2)) + r**2))]

    return simulate_system(equations_of_motion, initial_conditions, t, (vo, Rc, q))

# %%
# Generate the dataset
num_samples = 100  # Adjust as needed
subsample_size = 30

data_features = []
data_peak_freqs = []
data_potential_types = []

for potential_type, simulate_fn, num_params in [
    ("Harmonic Oscillator", simulate_harmonic_oscillator, 2),
    ("Modified Newtonian", simulate_modified_newtonian, 2),
    ("Custom Potential", simulate_custom_potential, 3),
    ("Disk Galaxy", simulate_disk_galaxy, 3),
    ("Another Disk Galaxy", simulate_another_disk_galaxy, 3)
]:
    for _ in range(num_samples):
        initial_conditions = np.random.rand(4) * 2 - 1
        params = np.random.rand(num_params)
        t_max = np.random.uniform(5, 20)
        t = np.linspace(0, t_max, 1000)

        if num_params == 2:
            t, x, y, v_x, v_y = simulate_fn(initial_conditions, t, params[0], params[1])
        elif num_params == 3:
            t, x, y, v_x, v_y = simulate_fn(initial_conditions, t, params[0], params[1], params[2])
        # Add more cases if needed

        t_sub, x_sub, y_sub, v_x_sub, v_y_sub = subsample_time_series(t, x, y, v_x, v_y, subsample_size)
        freqs, power = compute_lomb_scargle(t_sub, x_sub, y_sub)
        peak_freq = find_peak_frequency(freqs, power)
        features = extract_features(x_sub, y_sub, v_x_sub, v_y_sub)

        data_features.append(features)
        data_peak_freqs.append(peak_freq)
        data_potential_types.append(potential_type)

features_df = pd.DataFrame(data_features)
features_df['PeakFrequency'] = data_peak_freqs
features_df['PotentialType'] = data_potential_types


# %%
features_df

# %%



