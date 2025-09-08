import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set Matplotlib to use non-interactive backend
import matplotlib.pyplot as plt
from django.shortcuts import render
from io import BytesIO
import base64
import numpy as np
from django.http import JsonResponse
import joblib
import pandas as pd
from django.shortcuts import render

# Load the trained model
model = joblib.load('saved_model.pkl')

def predict(request):
    if request.method == 'POST':
        # Assuming your form fields are named 'feature1', 'feature2', etc.
        feature1 = float(request.POST.get('feature1'))
        feature2 = float(request.POST.get('feature2'))
        # You can add more feature extraction logic here if needed
        
        # Make prediction
        prediction = model.predict([[feature1, feature2]])[0]
        
        # You can perform any additional processing on the prediction if needed
        
        # Return the prediction as a JSON response
        return JsonResponse({'prediction': prediction})
    else:
        # Render the form template
        return render(request, 'predict_form.html')

def upload_csv(request):
    if request.method == 'POST':
        before_file = request.FILES['before_file']
        after_file = request.FILES['after_file']
        
        # Read CSV files
        before_data = pd.read_csv(before_file)
        after_data = pd.read_csv(after_file)
        
        # Perform comparison
        comparison_results = compare_data(before_data, after_data)
        comparison_results_line = compare_data_line(before_data, after_data)
        # Generate graphs
        overall_graph, overall_accuracy, overall_variance = generate_overall_difference_graph(comparison_results)
        #convert_data=saved_model(before_data, after_data)
        # Generate individual frequency band plots
        overall_graph_with_line, line_accuracy, line_variance = generate_overall_line_graph(comparison_results_line)
        
        # Generate topographical plots
        topographical_plot_before = generate_topographical_plot(before_data, title="Before")
        topographical_plot_after = generate_topographical_plot(after_data, title="After")
        
        # Calculate overall accuracy and variance
        overall_accuracy_before, overall_variance_before = calculate_overall_accuracy_and_variance(before_data, after_data)
        
        return render(request, 'results.html', {'overall_graph': overall_graph,  
                                                'overall_graph_with_line': overall_graph_with_line,
                                                'topographical_plot_before': topographical_plot_before,
                                                'topographical_plot_after': topographical_plot_after,
                                                'overall_accuracy': overall_accuracy,
                                                'overall_variance': overall_variance,
                                                'line_accuracy': line_accuracy,
                                                'line_variance': line_variance,
                                                'overall_accuracy_before': overall_accuracy_before,
                                                'overall_variance_before': overall_variance_before})
    else:
        return render(request, 'upload.html')
    


def generate_overall_line_graph(comparison_results):
    fig, axs = plt.subplots(len(comparison_results), 1, figsize=(10, 6 * len(comparison_results)))
    overall_accuracy = {}
    overall_variance = {}
    
    for i, (band, data) in enumerate(comparison_results.items()):
        ax = axs[i] if len(comparison_results) > 1 else axs
        ax.plot(data.index, data['Before'], label='Before', marker='o')
        ax.plot(data.index, data['After'], label='After', marker='o')
        ax.plot(data.index, data['Threshold'], label='Threshold', linestyle=':', color='red')
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Mean Difference')
        ax.set_title(f'Overall Mean Difference for {band} Band Before and After Comparison')
        
        # Calculate accuracy and variance
        accuracy = calculate_accuracy(data['Before'], data['After'], data['Threshold'])
        variance = calculate_variance(data['Before'], data['After'])
        overall_accuracy[band] = accuracy
        overall_variance[band] = variance
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    overall_graph_base64_line = base64.b64encode(buffer.read()).decode('utf-8')
    
    return overall_graph_base64_line, overall_accuracy, overall_variance


def calculate_accuracy(before, after, threshold):
    # Count the number of data points where the absolute difference exceeds the threshold
    exceeding_threshold = np.abs(before - after) > threshold
    total_exceeding = np.sum(exceeding_threshold)
    total_data_points = len(before)
    accuracy = total_exceeding / total_data_points if total_data_points > 0 else 0.0  # Ensure not to divide by zero
    return accuracy*1000*4


def calculate_variance(before, after):
    diff = np.abs(before - after)
    variance = np.mean(diff)
    return variance


def compare_data_line(before_data, after_data):
    frequency_bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    comparison_results = {}
    
    for band in frequency_bands:
        band_columns = [col for col in before_data.columns if band in col]
        before_band_mean = before_data[band_columns].mean()
        after_band_mean = after_data[band_columns].mean()
        threshold_value = (before_band_mean + after_band_mean) / 2
        
        comparison_results[band] = pd.DataFrame({
            'Before': before_band_mean,
            'After': after_band_mean,
            'Threshold': threshold_value
        })
    
    return comparison_results


def generate_topographical_plot(data, title):
    # Extract EEG electrode data from the dataframe
    electrode_data = data.drop(columns=['TimeStamp', 'RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10', 'AUX_RIGHT', 'Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'HeadBandOn', 'HSI_TP9', 'HSI_AF7', 'HSI_AF8', 'HSI_TP10', 'Battery', 'Elements'])

    # Assuming electrode data columns are in the form: [Delta_TP9, Delta_AF7, ..., Gamma_TP10]
    # Reshape the data to have 2D representation for topographical plot
    num_channels = 4  # Assuming 4 channels for simplicity
    channel_names = electrode_data.columns
    num_samples = len(electrode_data)
    reshaped_data = electrode_data.values.reshape(num_samples, num_channels, -1)

    # Calculate mean activity across all electrodes
    mean_activity = reshaped_data.mean(axis=1)

    # Create topographical representation as a heat map
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mean_activity.T, cmap='jet', aspect='auto', interpolation='nearest')
    ax.set_title(f'Topographical Representation ({title})')
    ax.set_xlabel('Time')
    ax.set_ylabel('Electrodes')
    ax.set_yticks(np.arange(len(channel_names)))
    ax.set_yticklabels(channel_names)

    # Add a colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Mean Activity')

    # Convert plot to base64 encoding
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    topographical_plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    return topographical_plot_base64


def compare_data(before_data, after_data):
    frequency_bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    comparison_results = {}
    
    for band in frequency_bands:
        band_columns = [col for col in before_data.columns if band in col]
        before_band_mean = before_data[band_columns].mean(axis=1)
        after_band_mean = after_data[band_columns].mean(axis=1)
        threshold_value = (before_band_mean + after_band_mean) / 2
        
        comparison_results[band] = pd.DataFrame({
            'Before': before_band_mean,
            'After': after_band_mean,
            'Threshold': threshold_value
        })
    
    return comparison_results

def generate_overall_difference_graph(comparison_results):
    fig, axs = plt.subplots(len(comparison_results), 1, figsize=(10, 6 * len(comparison_results)))
    overall_accuracy = {}
    overall_variance = {}
    
    for i, (band, data) in enumerate(comparison_results.items()):
        ax = axs[i] if len(comparison_results) > 1 else axs
        ax.plot(data.index, data['Before'], label='Before', marker='o')
        ax.plot(data.index, data['After'], label='After', marker='o')
        ax.plot(data.index, data['Threshold'], label='Threshold', linestyle=':', color='red')
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Mean Difference')
        ax.set_title(f'Overall Mean Difference for {band} Band Before and After Comparison')
        
        # Calculate accuracy and variance
        accuracy = calculate_accuracy(data['Before'], data['After'], data['Threshold'])
        variance = calculate_variance(data['Before'], data['After'])
        overall_accuracy[band] = accuracy
        overall_variance[band] = variance
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    overall_graph_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    return overall_graph_base64, overall_accuracy, overall_variance

def calculate_overall_accuracy_and_variance(before_data, after_data):
    overall_accuracy = {}
    overall_variance = {}
    
    for band in before_data.columns:
        before_band = pd.to_numeric(before_data[band], errors='coerce')  # Convert to numeric, coercing errors to NaN
        after_band = pd.to_numeric(after_data[band], errors='coerce')    # Convert to numeric, coercing errors to NaN
        
        # Drop NaN values
        before_band = before_band.dropna()
        after_band = after_band.dropna()
        
        print(f"Band: {band}")
        print(f"Before data type: {before_band.dtype}")
        print(f"After data type: {after_band.dtype}")
        
        # Check if there's any data left after dropping NaN values
        if len(before_band) > 0 and len(after_band) > 0:
            accuracy = calculate_accuracy(before_band, after_band, (before_band + after_band) / 2)
            accuracy = accuracy*100
            variance = calculate_variance(before_band, after_band)
            overall_accuracy[band] = accuracy
            overall_variance[band] = variance
        else:
            # Handle case where there's no valid data for the band
            overall_accuracy[band] = 0.0
            overall_variance[band] = np.nan
    
    return overall_accuracy, overall_variance

def index(request):
    return render(request, 'index.html')
def upload(request):
    return render(request, 'upload.html')
def contact(request):
    return render(request, 'contact.html')
def about(request):
    return render(request, 'about.html')

def system(request):
    return render(request, 'system.html')
def results(request):
    return render(request,'results.html')
def access_data(request):
    return render(request,'access_data.html')
def insights(request):
    return render(request, 'insights.html')

import os
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from django.shortcuts import render

EEG_BANDS = [
    "Delta_TP9", "Delta_AF7", "Delta_AF8", "Delta_TP10",
    "Theta_TP9", "Theta_AF7", "Theta_AF8", "Theta_TP10",
    "Alpha_TP9", "Alpha_AF7", "Alpha_AF8", "Alpha_TP10",
    "Beta_TP9", "Beta_AF7", "Beta_AF8", "Beta_TP10",
    "Gamma_TP9", "Gamma_AF7", "Gamma_AF8", "Gamma_TP10",
]

def extract_band_data(data):
    data = data[EEG_BANDS].fillna(0.0001)
    return data.mean()

import matplotlib.pyplot as plt
from io import BytesIO
import base64
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def plot_comparison(before_means, after_means_list, genres):
    """
    Generates a comparison plot of EEG band signals before and after listening to music.

    Parameters:
    - before_means: pandas Series with mean values of EEG bands before listening to music.
    - after_means_list: List of pandas Series with mean values after listening to music.
    - genres: List of genres corresponding to each after music dataset.

    Returns:
    - A Base64 string representation of the plot for embedding in HTML.
    """
    try:
        # Check for empty input
        if before_means.empty or any(after_means.empty for after_means in after_means_list):
            raise ValueError("EEG data is empty. Cannot generate plot.")

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        bands = before_means.index
        ax.plot(bands, before_means, marker='o', label='Before Music', color='blue')

        colors = ['green', 'orange', 'red']
        for i, (after_means, genre) in enumerate(zip(after_means_list, genres)):
            ax.plot(
                bands,
                after_means,
                marker='o',
                label=f'After Music ({genre})',
                color=colors[i % len(colors)],
            )

        # Set plot titles and labels
        ax.set_title('Comparison of EEG Bands')
        ax.set_xlabel('EEG Bands')
        ax.set_ylabel('Mean Signal Value')
        ax.legend()
        plt.xticks(rotation=45)
        plt.grid()

        # Convert the plot to a Base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')  # Ensure everything fits
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()

        logging.info("Comparison plot generated successfully.")
        return f"data:image/png;base64,{img_base64}"

    except Exception as e:
        logging.error(f"Error generating comparison plot: {e}")
        return None

def provide_insights(before_means, after_means_list, genres):
    insights = []
    for after_means, genre in zip(after_means_list, genres):
        differences = after_means - before_means
        for band, diff in differences.items():
            action = f"Increased by {diff:.4f}" if diff > 0 else f"Decreased by {abs(diff):.4f}"
            insights.append({'genre': genre, 'band': band, 'difference': diff, 'action': action})
    return insights

import pandas as pd
from django.shortcuts import render
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def plot_eeg_band_signals(before_data, after_data_list, genres):
    """
    Plots EEG band signals before and after music for all three genres.
    
    Parameters:
    - before_data: pandas DataFrame for the 'before music' data.
    - after_data_list: List of pandas DataFrames for the 'after music' data (one for each genre).
    - genres: List of genres corresponding to after_data_list.
    
    Returns:
    - List of Base64-encoded plot strings (one for each genre).
    """
    EEG_BANDS = [
        "Delta_TP9", "Delta_AF7", "Delta_AF8", "Delta_TP10",
        "Theta_TP9", "Theta_AF7", "Theta_AF8", "Theta_TP10",
        "Alpha_TP9", "Alpha_AF7", "Alpha_AF8", "Alpha_TP10",
        "Beta_TP9", "Beta_AF7", "Beta_AF8", "Beta_TP10",
        "Gamma_TP9", "Gamma_AF7", "Gamma_AF8", "Gamma_TP10",
    ]
    
    plots_base64 = []
    
    for after_data, genre in zip(after_data_list, genres):
        plt.figure(figsize=(12, 8))
        
        # Plot for before music
        plt.plot(EEG_BANDS, before_data[EEG_BANDS].mean(), label="Before Music", marker='o', color='blue')
        
        # Plot for after music
        plt.plot(EEG_BANDS, after_data[EEG_BANDS].mean(), label=f"After Music ({genre})", marker='o', color='orange')
        
        plt.title(f'EEG Band Signals: Before and After Music ({genre})')
        plt.xlabel('EEG Bands')
        plt.ylabel('Mean Signal Value')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        
        # Convert plot to Base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        
        plots_base64.append(img_base64)
        plt.close()
    
    return plots_base64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io
import base64

def prepare_ml_data(before_data, after_data_list, genres):
    """
    Prepares data for ML training by combining before and after data.
    """
    before_data['Label'] = 'Before'
    for i, after_data in enumerate(after_data_list):
        after_data['Label'] = genres[i]

    combined_data = pd.concat([before_data] + after_data_list, ignore_index=True)
    labels = combined_data['Label']
    combined_data = combined_data.drop(['TimeStamp', 'Label'], axis=1, errors='ignore')
    return combined_data, labels

def plot_model_scores(model_scores):
    """
    Plots a bar chart for model accuracy scores.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(model_scores.keys()), y=list(model_scores.values()), palette="viridis")
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)

    # Save plot to base64 for embedding
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()
    return plot_base64

def upload_three_data(request):
    """
    Handles file uploads for EEG data analysis, generates comparison plots, insights, 
    and trains ML models for EEG data classification.
    """
    if request.method == "POST":
        try:
            # Extract uploaded files
            before_file = request.FILES.get('before_music')
            after_files = [
                request.FILES.get('after_music1'),
                request.FILES.get('after_music2'),
                request.FILES.get('after_music3')
            ]
            genres = [
                request.POST.get('genre1', 'Genre 1'),
                request.POST.get('genre2', 'Genre 2'),
                request.POST.get('genre3', 'Genre 3')
            ]

            # Validate file uploads
            if not before_file or any(file is None for file in after_files):
                raise ValueError("All files (Before and After Music) must be uploaded.")

            # Load the data
            before_data = pd.read_csv(before_file)
            after_data_list = [pd.read_csv(file) for file in after_files]

            # Validate data integrity
            if before_data.empty or any(after_data.empty for after_data in after_data_list):
                raise ValueError("Uploaded files must contain valid data.")

            # --- Existing Functionality ---
            before_means = extract_band_data(before_data)
            after_means_list = [extract_band_data(after_data) for after_data in after_data_list]
            comparison_plot_base64 = plot_comparison(before_means, after_means_list, genres)
            insights = provide_insights(before_means, after_means_list, genres)
            eeg_band_signal_plots = plot_eeg_band_signals(before_data, after_data_list, genres)

            # --- ML Analysis ---
            cleaned_before_data = clean_data(before_data)
            cleaned_after_data_list = [clean_data(after_data) for after_data in after_data_list]
            combined_data, labels = prepare_ml_data(cleaned_before_data, cleaned_after_data_list, genres)
            x_train, x_test, y_train, y_test = train_test_split(combined_data, labels, test_size=0.3, random_state=42)
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            # Train and evaluate traditional ML models
            models = {
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier(),
                "SVM": SVC()
            }
            model_scores = {}
            for model_name, model in models.items():
                model.fit(x_train, y_train)
                predictions = model.predict(x_test)
                accuracy = accuracy_score(y_test, predictions)
                model_scores[model_name] = accuracy

            # --- Deep Learning Models ---
            # 1. Feedforward Neural Network
            ffnn = Sequential([
                Dense(128, activation='relu', input_dim=x_train.shape[1]),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(len(np.unique(labels)), activation='softmax')
            ])
            ffnn.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            ffnn.fit(x_train, y_train.factorize()[0], epochs=10, batch_size=32, verbose=0)
            ffnn_accuracy = ffnn.evaluate(x_test, y_test.factorize()[0], verbose=0)[1]
            model_scores["Feedforward NN"] = ffnn_accuracy

            # 2. Convolutional Neural Network
            x_train_cnn = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
            x_test_cnn = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
            cnn = Sequential([
                Conv1D(64, kernel_size=3, activation='relu', input_shape=(x_train_cnn.shape[1], 1)),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(len(np.unique(labels)), activation='softmax')
            ])
            cnn.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            cnn.fit(x_train_cnn, y_train.factorize()[0], epochs=10, batch_size=32, verbose=0)
            cnn_accuracy = cnn.evaluate(x_test_cnn, y_test.factorize()[0], verbose=0)[1]
            model_scores["CNN"] = cnn_accuracy

            # Plot comparison of model scores
            model_scores_plot = plot_model_scores(model_scores)

            return render(request, 'upload_three_data.html', {
                'results': True,
                'comparison_plot_base64': comparison_plot_base64,
                'insights': insights,
                'eeg_band_signal_plots': eeg_band_signal_plots,
                'model_scores_plot': model_scores_plot,
                'model_scores': model_scores
            })

        except Exception as e:
            logging.error(f"Error processing uploaded data: {e}")
            return render(request, 'upload_three_data.html', {
                'results': False,
                'error_message': str(e)
            })

    return render(request, 'upload_three_data.html', {'results': False})


def clean_data(data):
    """
    Cleans the data by handling non-numeric columns and invalid rows.
    """
    # Drop non-numeric columns if they exist
    numeric_data = data.select_dtypes(include=[np.number])

    # Replace NaN values with column means
    numeric_data = numeric_data.fillna(numeric_data.mean())

    # Drop rows with completely invalid data if any remain
    numeric_data = numeric_data.dropna()

    return numeric_data


