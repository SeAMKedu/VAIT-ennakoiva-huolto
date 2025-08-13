import pandas as pd  # Import pandas for data manipulation / Tuo pandas-datan käsittelyyn
import numpy as np   # Import numpy for numerical operations / Tuo numpy-laskentaan
from datetime import datetime, timedelta  # Import datetime utilities / Tuo datetime-työkalut
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier for machine learning / Tuo RandomForestClassifier koneoppimiseen
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score  # Import metrics for evaluation / Tuo mittarit mallin arviointiin
import matplotlib.pyplot as plt  # Import matplotlib for plotting / Tuo matplotlib-grafiikoiden tekemiseen
from matplotlib.dates import DateFormatter  # Import DateFormatter for date formatting / Tuo DateFormatter päivämäärien muotoiluun
import matplotlib.dates as mdates  # Import matplotlib.dates for handling date locators / Tuo matplotlib.dates päivämääräajan hallintaan

#Euroopan Unionin osarahoittama. 
#Luotu vAI:lla tuottavuutta? -hankkeessa osana yhtä tuottavuuspilottia. 
#https://projektit.seamk.fi/alykkaat-teknologiat/vailla-tuottavuutta/

# --- READ THE CSV FILE / LUETAAN CSV-TIEDOSTO ---
try:
    # Read CSV file and automatically parse the 'time' column as dates
    # Luetaan CSV-tiedosto ja jäsennetään 'time'-sarake automaattisesti päivämääriksi
    # Tässä tiedosto on vibration_data_kaikki.csv
    df = pd.read_csv('vibration_data_kaikki.csv', parse_dates=['time'])
except FileNotFoundError:
    # Handle the case where the file does not exist / Käsitellään tilanne, jos tiedostoa ei löydy
    print("Error: vibration_data_kaikki.csv not found. Please ensure the file exists in the same directory.")
    exit()
except Exception as e:
    # Handle any other exceptions during file reading / Käsitellään muita virheitä tiedoston lukemisessa
    print(f"An error occurred while reading the CSV file: {e}")
    exit()

# --- SORT DATA BY TIMESTAMP / LAJITELLAAN DATA AIKALEIMEN MUKAAN ---
# Ensure the data is in chronological order / Varmistetaan, että data on aikajärjestyksessä
df.sort_values(by='time', inplace=True)

# --- GROUP DATA INTO TIME WINDOWS AND CALCULATE STATISTICS / RYHMITETÄÄN DATA AIKAVÄLIIN JA LASKETAAN TILASTOT ---
window_size = '1T'  # Define window size as 1 minute / Määritellään ikkunan kooksi 1 minuutti
grouped = df.set_index('time').resample(window_size)  # Resample data based on time window / Resamplataan data ajan perusteella

# Calculate statistics (mean, std, min, max, count) for x, and similar for y and z axes
# Lasketaan tilastollisia arvoja (keskiarvo, keskihajonta, min, max, lukumäärä) x-akselille ja vastaavasti y- ja z-akselille
window_stats = grouped.agg({
    'x': ['mean', 'std', 'min', 'max', 'count'],
    'y': ['mean', 'std', 'min', 'max'],
    'z': ['mean', 'std', 'min', 'max']
}).dropna()  # Remove windows with missing data / Poistetaan ikkunat, joissa puuttuu dataa

# Filter out time windows with no data (count of x must be greater than 0)
# Suodatetaan pois ne ajanjaksot, joissa ei ole dataa (x:n lukumäärän on oltava yli 0)
valid_window_stats = window_stats[window_stats[('x', 'count')] > 0]

# Print summary information about the time windows / Tulostetaan yhteenvetotietoa aikaväleistä
print(f"Total time windows: {len(window_stats)}")
print(f"Windows with data: {len(valid_window_stats)}")
print(f"First window timestamp: {valid_window_stats.index[0]}")
print(f"Last window timestamp: {valid_window_stats.index[-1]}")

# --- DEFINE KNOWN BLADE CHANGE TIMES FOR EVALUATION / MÄÄRITELLÄÄN TUNNETUT TERÄN VAIHTOAJAT ARVIOINTIA VARTEN ---
# Each tuple holds one or more known blade change times (for evaluation purposes)
# Jokainen tuple sisältää yhden tai useamman tunnetun lterämuutoksen ajan (vain arviointiin)
part_changes = [
    ('2025-02-17 06:30:00',),
    ('2025-02-18 01:00:00', '2025-02-18 10:00:00'),
    ('2025-02-19 09:00:00', '2025-02-19 14:00:00'),
    ('2025-02-20 11:30:00',),
    ('2025-03-03 12:00:00', '2025-03-03 14:30:00', '2025-03-03 20:30:00'),
    ('2025-03-04 14:00:00',),
    ('2025-03-05 11:30:00', '2025-03-05 15:30:00', '2025-03-05 16:30:00'),
    ('2025-03-06 10:00:00', '2025-03-06 15:30:00', '2025-03-06 20:30:00'),
    ('2025-03-07 07:00:00', '2025-03-07 14:00:00'),
    ('2025-03-11 02:00:00', '2025-03-11 12:00:00'),
    ('2025-03-12 06:00:00', '2025-03-12 10:30:00', '2025-03-12 12:30:00', '2025-03-12 13:30:00'),
    ('2025-03-13 10:00:00', '2025-03-13 11:00:00', '2025-03-13 21:00:00', '2025-03-13 23:00:00'),
    ('2025-03-14 06:00:00', '2025-03-14 07:00:00')
]

# Flatten the part_changes list and convert each time to a datetime object with UTC timezone if not already set
# Tasoitetaan part_changes ja muunnetaan jokainen aika datetime-muotoon, asetetaan UTC-aikavyöhyke jos puuttuu
known_changes_flat = []
for changes in part_changes:
    for change_time in changes:
        change_dt = pd.to_datetime(change_time)
        if change_dt.tzinfo is None:
            change_dt = change_dt.tz_localize('UTC')
        known_changes_flat.append(change_dt)

# --- SPLIT DATA INTO TRAINING AND TEST SETS BASED ON TIME / JAETAAN DATA AIKAJÄRJESTYKSEEN PERUSTUVINEN KOULUTUS- JA TESTIAINEISTO ---
data_timeline = valid_window_stats.index  # Get the timeline of windows / Haetaan aikajanalla ikkunat
split_idx = int(len(data_timeline) * 0.7)   # Use 70% of the data for training / Käytetään 70 % datasta koulutukseen
train_end_time = data_timeline[split_idx]     # Determine the end time for the training data / Määritellään koulutusdatan loppuaika

# Split the data into training (earlier portion) and testing (later portion simulating unseen future data)
# Jaetaan data: koulutusaineisto (alkupään data) ja testiaineisto (jälkimmäinen osa "tulevaa" dataa varten)
train_data = valid_window_stats[valid_window_stats.index <= train_end_time].copy()
test_data = valid_window_stats[valid_window_stats.index > train_end_time].copy()

# --- FEATURE PREPARATION FUNCTION / OMINAISUUKSIEN VALMISTELUFUNKTIO ---
def prepare_features(df_original):
    # Create a copy of the original dataframe
    # Luodaan kopio alkuperäisestä DataFramesta
    df = df_original.copy()

    # Create lag features to capture previous standard deviations and means over previous windows
    # Luodaan viiveominaisuuksia, jotka tallentavat edellisten ikkunoiden keskihajonnat ja keskiarvot
    for i in range(1, 4):
        df[f'lag_{i}_x_std'] = df[('x', 'std')].shift(i)
        df[f'lag_{i}_y_std'] = df[('y', 'std')].shift(i)
        df[f'lag_{i}_z_std'] = df[('z', 'std')].shift(i)
        df[f'lag_{i}_x_mean'] = df[('x', 'mean')].shift(i)
        df[f'lag_{i}_y_mean'] = df[('y', 'mean')].shift(i)
        df[f'lag_{i}_z_mean'] = df[('z', 'mean')].shift(i)

    # Calculate rolling statistics for additional context over different window sizes (2 and 3)
    # Lasketaan liukuvia tilastoja lisäyhteyksien saamiseksi eri ikkunoilla (2 ja 3)
    for window in [2, 3]:
        df[f'roll_{window}_x_std'] = df[('x', 'std')].rolling(window).std()
        df[f'roll_{window}_y_std'] = df[('y', 'std')].rolling(window).std()
        df[f'roll_{window}_z_std'] = df[('z', 'std')].rolling(window).std()
        df[f'roll_{window}_x_mean'] = df[('x', 'mean')].rolling(window).mean()
        df[f'roll_{window}_y_mean'] = df[('y', 'mean')].rolling(window).mean()
        df[f'roll_{window}_z_mean'] = df[('z', 'mean')].rolling(window).mean()
        df[f'roll_{window}_x_range'] = df[('x', 'max')].rolling(window).max() / (df[('x', 'min')].rolling(window).min() + 1e-6)
        df[f'roll_{window}_y_range'] = df[('y', 'max')].rolling(window).max() / (df[('y', 'min')].rolling(window).min() + 1e-6)
        df[f'roll_{window}_z_range'] = df[('z', 'max')].rolling(window).max() / (df[('z', 'min')].rolling(window).min() + 1e-6)

    # Calculate the rate of change (difference ratio) in the standard deviation for each axis
    # Lasketaan muutoksen nopeus (muutos-suhde) standardipoikkeamille jokaiselle akselille
    df['x_std_roc'] = df[('x', 'std')].diff() / df[('x', 'std')].shift(1).replace(0, np.nan)
    df['y_std_roc'] = df[('y', 'std')].diff() / df[('y', 'std')].shift(1).replace(0, np.nan)
    df['z_std_roc'] = df[('z', 'std')].diff() / df[('z', 'std')].shift(1).replace(0, np.nan)

    # Add time-based features: hour of the day and day of the week from the index
    # Lisätään aikaominaisuuksia: kellonaika ja viikonpäivä indeksistä
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek

    # Clean up any infinite or missing values from calculations
    # Poistetaan äärelliset ja puuttuvat arvot laskelmista
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    return df

# --- CREATE TARGET VARIABLE / LUODAAN TAVOITEMUUTTUJA ---
def create_target_variable(df, known_changes, prediction_window=timedelta(hours=2)):
    # Create a copy of the dataframe and initialize the target column for blade change as 0
    # Luodaan kopio DataFramesta ja alustetaan tavoitesarake terämuutoksille nollilla
    df = df.copy()
    df['blade_change'] = 0

    # For each known change, mark the time window before the change (prediction_window) with 1
    # Jokaiselle tunnetulle muutokselle merkitään ajanjakso ennen muutosta (prediction_window) arvolla 1
    for change_time in known_changes:
        window_start = change_time - prediction_window
        mask = (df.index >= window_start) & (df.index <= change_time)
        df.loc[mask, 'blade_change'] = 1

    return df

# Filter the known change times to only include those within the training period
# Suodatetaan tunnetut muutokset niin, että vain koulutusajanjaksoon kuuluvat säilytetään
train_changes = [change for change in known_changes_flat if change <= train_end_time]

# Create target variables for the training data using the known changes and then prepare features
# Luodaan tavoitemuuttujat koulutusdatasta ja valmistellaan ominaisuudet
train_data_with_targets = create_target_variable(train_data, train_changes)
prepared_train_data = prepare_features(train_data_with_targets)

# --- SPLIT FEATURES AND TARGET / JAETAAN OMINAISUUDET JA TAVOITE ---
X_train = prepared_train_data.drop(columns=['blade_change'])
# Drop columns that are not used for training (like count, min, max values)
# Poistetaan sarakkeet, joita ei käytetä koulutuksessa (esimerkiksi count, min, max-arvot)
X_train = X_train.loc[:, ~X_train.columns.isin([('x', 'count'), ('x', 'min'), ('x', 'max'),
                                                 ('y', 'min'), ('y', 'max'),
                                                 ('z', 'min'), ('z', 'max')])]
y_train = prepared_train_data['blade_change']

# --- TRAIN A CLASSIFIER / KOULUTETAAN LUOKITTELIJA ---
# Initialize and train a RandomForestClassifier using the prepared training data
# Alustetaan ja koulutetaan RandomForestClassifier käyttämällä valmisteltua koulutusdataa
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# --- FEATURE IMPORTANCE ANALYSIS / OMINAISUUKSIEN TÄRKEYDEN ANALYYSI ---
# Create a DataFrame to show the importance of each feature sorted in descending order
# Luodaan DataFrame, joka näyttää kunkin ominaisuuden tärkeyden laskevassa järjestyksessä
feature_importances = pd.DataFrame(
    model.feature_importances_,
    index=X_train.columns,
    columns=['importance']
).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importances.head(10))

# --- PREPARE TEST DATA / VALMISTELLAAN TESTIDATA ---
# Prepare the test data using the same feature preparation function (without known blade changes)
# Valmistellaan testidata käyttäen samaa ominaisuuksien valmistelufunktiota (ilman tunnettuja terämuutoksia)
prepared_test_data = prepare_features(test_data)
# Ensure the test features match the training set columns
# Varmistetaan, että testin ominaisuudet vastaavat koulutusaineiston sarakkeita
X_test = prepared_test_data.loc[:, X_train.columns]

# --- MAKE PREDICTIONS / TEHDÄN ENNUSTEITA ---
# Predict probabilities for blade change from the test data
# Ennustetaan todennäköisyydet terämuutokselle testidatassa
y_pred_proba = model.predict_proba(X_test)[:, 1]
# Convert probabilities to binary predictions using a threshold of 0.07 (high confidence)
# Muutetaan todennäköisyydet binaarisiksi ennusteiksi käyttäen kynnysarvoa 0.07
y_pred = (y_pred_proba >= 0.07).astype(int)

# Create a DataFrame for predictions with corresponding timestamps
# Luodaan DataFrame ennusteille, joissa on mukana aikaleimat
predictions_df = pd.DataFrame({
    'timestamp': prepared_test_data.index,
    'probability': y_pred_proba,
    'prediction': y_pred
})

# --- IDENTIFY PREDICTED BLADE CHANGE EVENTS / TUNNISTETAAN ENNUSTETUT teräMUUTOSTAPAHTUMAT ---
def identify_blade_change_events(predictions, threshold=0.10, cooldown_period=6):
    # Function to identify events from consecutive windows that exceed a probability threshold,
    # ensuring a minimum cooldown period between events.
    # Funktio tunnistamaan tapahtumia peräkkäisistä ikkunoista, jotka ylittävät kynnysarvon, varmistaen
    # minimiaikavälin (cooldown) tapahtumien välillä.
    events = []
    last_event_idx = -cooldown_period - 1  # Initialize with a value ensuring first event is recorded

    for idx, (timestamp, prob) in enumerate(zip(predictions['timestamp'], predictions['probability'])):
        if prob >= threshold and (idx - last_event_idx) > cooldown_period:
            events.append(timestamp)
            last_event_idx = idx

    return events

# Identify predicted blade change events from the predictions DataFrame
# Tunnistetaan ennustetut terämuutostapahtumat ennusteiden DataFramesta
predicted_events = identify_blade_change_events(predictions_df)
print(f"\nPredicted {len(predicted_events)} blade change events in test period")

# --- EVALUATE PREDICTIONS / ARVIOIDAAN ENNUSTEITA ---
# Filter known change times for the test period (after training data)
# Suodatetaan tunnetut terämuutokset vain testiaineiston ajalle (koulutuksen jälkeen)
test_changes = [change for change in known_changes_flat if change > train_end_time]

# Function to evaluate predictions by comparing predicted events with actual events within a given time window
# Funktio ennusteiden arvioimiseen vertaamalla ennustettuja tapahtumia todellisiin tapahtumiin määritellyssä aikavälissä
def evaluate_predictions(predicted_events, actual_events, time_window=timedelta(hours=1)):
    true_positives = []
    matched_actuals = set()

    for pred_time in predicted_events:
        matched = False
        for actual_time in actual_events:
            if abs(pred_time - actual_time) <= time_window:
                true_positives.append((pred_time, actual_time))
                matched_actuals.add(actual_time)
                matched = True
                break
        if not matched:
            continue

    missed_events = [actual for actual in actual_events if actual not in matched_actuals]
    false_positives = [pred for pred in predicted_events if not any(abs(pred - actual) <= time_window for actual in actual_events)]

    return true_positives, false_positives, missed_events

# Evaluate our predictions against the test changes
# Arvioidaan ennusteita testiaineiston todellisten muutosten perusteella
true_positives, false_positives, missed_events = evaluate_predictions(predicted_events, test_changes)

print(f"\nEvaluation Results (1-hour window):")
print(f"True Positives: {len(true_positives)} predictions")
print(f"False Positives: {len(false_positives)} predictions")
print(f"Missed Events: {len(missed_events)} actual changes")

if len(test_changes) > 0:
    recall = len(true_positives) / len(test_changes)
    precision = len(true_positives) / (len(true_positives) + len(false_positives)) if (len(true_positives) + len(false_positives)) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Recall: {recall:.2f} (percentage of actual changes detected)")
    print(f"Precision: {precision:.2f} (percentage of predictions that were correct)")
    print(f"F1 Score: {f1:.2f} (harmonic mean of precision and recall)")

# --- PLOTTING SECTION / GRAFIIKKOJEN PIIRTAMINEN ---
# Define colorblind-friendly colors for visualizing different elements
# Määritellään värimaailma, joka sopii värisokeille (erilaisten elementtien visualisointia varten)
colors = {
    'x_std': '#1f77b4',      # Blue for X-axis standard deviation / Sininen X-akselin keskihajonnalle
    'y_std': '#ff7f0e',      # Orange for Y-axis standard deviation / Oranssi Y-akselin keskihajonnalle
    'z_std': '#2ca02c',      # Green for Z-axis standard deviation / Vihreä Z-akselin keskihajonnalle
    'probability': '#9467bd',# Purple for blade change probability / Violetti terämuutoksen todennäköisyydelle
    'actual': '#d62728',     # Red for actual blade change / Punainen todelliselle terämuutokselle
    'predicted': '#8c564b'   # Brown for predicted blade change / Ruskea ennustetulle terämuutokselle
}

plt.figure(figsize=(15, 8))

# Plot the standard deviation of vibration data for each axis in the test set
# Piirretään testidatan tärinädata: X, Y ja Z-akselien keskihajonnat
ax1 = plt.gca()
ax1.plot(prepared_test_data.index, prepared_test_data[('x', 'std')], color=colors['x_std'], alpha=0.5, label='X Std Dev')
ax1.plot(prepared_test_data.index, prepared_test_data[('y', 'std')], color=colors['y_std'], alpha=0.5, label='Y Std Dev')
ax1.plot(prepared_test_data.index, prepared_test_data[('z', 'std')], color=colors['z_std'], alpha=0.5, label='Z Std Dev')
ax1.set_ylabel('Vibration Std Dev', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_ylim(bottom=0)

# Create a second y-axis to plot the probability of blade change
# Luodaan toinen y-akseli terämuutoksen todennäköisyyden piirtämiseen
ax2 = ax1.twinx()
ax2.plot(predictions_df['timestamp'], predictions_df['probability'], color=colors['probability'], label='Blade Change Probability')
ax2.set_ylabel('Probability', color=colors['probability'])
ax2.tick_params(axis='y', labelcolor=colors['probability'])
ax2.set_ylim([0, 1])

# Mark actual blade changes (known changes in test period) with vertical dashed lines
# Merkitään todelliset terämuutokset (testiaineistoon kuuluvat) pystysuorilla katkoviivoilla
for actual in test_changes:
    plt.axvline(x=actual, color=colors['actual'], linestyle='--', alpha=0.7)

# Mark predicted blade changes with vertical solid lines
# Merkitään ennustetut terämuutokset pystysuorilla jatkuvilla viivoilla
for predicted in predicted_events:
    plt.axvline(x=predicted, color=colors['predicted'], linestyle='-', alpha=0.7)

# Create a custom legend for the plot using custom lines
# Luodaan mukautettu selite (legend) kuvalle käyttäen määriteltyjä viivoja
from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], color=colors['x_std'], lw=2, alpha=0.5),
    Line2D([0], [0], color=colors['y_std'], lw=2, alpha=0.5),
    Line2D([0], [0], color=colors['z_std'], lw=2, alpha=0.5),
    Line2D([0], [0], color=colors['probability'], lw=2),
    Line2D([0], [0], color=colors['actual'], linestyle='--', lw=2),
    Line2D([0], [0], color=colors['predicted'], lw=2)
]
ax1.legend(custom_lines, ['X Std Dev', 'Y Std Dev', 'Z Std Dev', 'Blade Change Probability',
                          'Actual Blade Change', 'Predicted Blade Change'],
           loc='upper left', bbox_to_anchor=(0, -0.15), ncol=3)

# Format the date axis for better readability
# Muotoillaan päivämääräakseli selkeäksi
plt.gcf().autofmt_xdate()
date_format = DateFormatter('%Y-%m-%d')
plt.gca().xaxis.set_major_formatter(date_format)
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))

plt.title('Blade Change Prediction - True Blind Prediction on Test Period')
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.show()

# --- PLOT AN EXAMPLE WINDOW AROUND A CORRECTLY PREDICTED BLADE CHANGE / PIIRRÄ ESIMERKKIAIKAJAKSO OIKEIN ENNUSTETUN teräMUUTOKSEN YMPÄRILLÄ ---
if true_positives:
    # Get the first correctly predicted event and its matching actual event
    # Haetaan ensimmäinen oikein ennustettu tapahtuma ja siihen vastaava todellinen tapahtuma
    pred_time, actual_time = true_positives[0]
    window_start = actual_time - timedelta(hours=6)
    window_end = actual_time + timedelta(hours=2)
    # Extract the data for the window around the event
    # Poimitaan data aikaväliltä, joka ympäröi tapahtumaa
    window_data = valid_window_stats[(valid_window_stats.index >= window_start) & (valid_window_stats.index <= window_end)]
    window_predictions = predictions_df[(predictions_df['timestamp'] >= window_start) & (predictions_df['timestamp'] <= window_end)]

    plt.figure(figsize=(12, 8))
    ax1 = plt.gca()
    ax1.plot(window_data.index, window_data[('x', 'std')], color=colors['x_std'], label='X Std Dev')
    ax1.plot(window_data.index, window_data[('y', 'std')], color=colors['y_std'], label='Y Std Dev')
    ax1.plot(window_data.index, window_data[('z', 'std')], color=colors['z_std'], label='Z Std Dev')
    ax1.set_ylabel('Vibration Std Dev')

    # Mark the actual and predicted events within the window
    # Merkitään ikkunassa sekä todellinen että ennustettu tapahtuma
    plt.axvline(x=actual_time, color=colors['actual'], linestyle='--', label='Actual Blade Change')
    plt.axvline(x=pred_time, color=colors['predicted'], linestyle='-', label='Predicted Blade Change')

    plt.gcf().autofmt_xdate()
    detailed_format = DateFormatter('%Y-%m-%d %H:%M')
    plt.gca().xaxis.set_major_formatter(detailed_format)
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))

    plt.title(f'Detailed View of a Correctly Predicted Blade Change')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
