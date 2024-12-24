import streamlit as st
import pickle
import pandas as pd

#loading the pickled model
model_filename = "tuned_random_forest_model.pkl"

@st.cache_resource
def load_model():
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

st.title("🩺 Disease Prediction App")
st.markdown("""
<style>
    body {
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border-radius: 12px;
        font-size: 18px;
    }
    .stButton:hover>button {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

st.write("Predict diseases based on symptoms using the trained Random Forest model.")

# Sidebar
st.sidebar.header("ℹ️ About")
st.sidebar.markdown("""
This app uses a machine learning model trained on a dataset of symptoms and diseases to predict the likely disease based on selected symptoms.
- **Model**: Random Forest Classifier
- **Accuracy**: ~94%
""")

# Loading CSV resource
@st.cache_resource
def load_data():
    diets = pd.read_csv("mapping\\Filtered_Diets_Dataset.csv")
    medications = pd.read_csv("mapping\\Filtered_Medications_Dataset.csv")
    precautions = pd.read_csv("mapping\\Filtered_Precautions_Dataset.csv")
    workouts = pd.read_csv("mapping\\Filtered_Workouts_Dataset.csv")
    return diets, medications, precautions, workouts

diets_df, medications_df, precautions_df, workouts_df = load_data()

# App Title
st.title("Disease Prediction and Prescription App")
st.write("Predict diseases based on symptoms and get recommendations for medications, diets, precautions, and workouts.")

st.header("Select Symptoms")

# List of symptoms
all_symptoms = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering", "chills", 
    "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting", "vomiting", 
    "burning_micturition", "spotting_urination", "fatigue", "weight_gain", "anxiety", 
    "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness", "lethargy", 
    "patches_in_throat", "irregular_sugar_level", "cough", "high_fever", "sunken_eyes", 
    "breathlessness", "sweating", "dehydration", "indigestion", "headache", "yellowish_skin", 
    "dark_urine", "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain", 
    "constipation", "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine", 
    "yellowing_of_eyes", "acute_liver_failure", "fluid_overload", "swelling_of_stomach", 
    "swelled_lymph_nodes", "malaise", "blurred_and_distorted_vision", "phlegm", 
    "throat_irritation", "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion", 
    "chest_pain", "weakness_in_limbs", "fast_heart_rate", "pain_during_bowel_movements", 
    "pain_in_anal_region", "bloody_stool", "irritation_in_anus", "neck_pain", "dizziness", 
    "cramps", "bruising", "obesity", "swollen_legs", "swollen_blood_vessels", 
    "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails", "swollen_extremeties", 
    "excessive_hunger", "extra_marital_contacts", "drying_and_tingling_lips", "slurred_speech", 
    "knee_pain", "hip_joint_pain", "muscle_weakness", "stiff_neck", "swelling_joints", 
    "movement_stiffness", "spinning_movements", "loss_of_balance", "unsteadiness", 
    "weakness_of_one_body_side", "loss_of_smell", "bladder_discomfort", "foul_smell_of_urine", 
    "continuous_feel_of_urine", "passage_of_gases", "internal_itching", "toxic_look_(typhos)", 
    "depression", "irritability", "muscle_pain", "altered_sensorium", "red_spots_over_body", 
    "belly_pain", "abnormal_menstruation", "dischromic_patches", "watering_from_eyes", 
    "increased_appetite", "polyuria", "family_history", "mucoid_sputum", "rusty_sputum", 
    "lack_of_concentration", "visual_disturbances", "receiving_blood_transfusion", 
    "receiving_unsterile_injections", "coma", "stomach_bleeding", "distention_of_abdomen", 
    "history_of_alcohol_consumption", "fluid_overload_1", "blood_in_sputum", 
    "prominent_veins_on_calf", "palpitations", "painful_walking", "pus_filled_pimples", 
    "blackheads", "scurring", "skin_peeling", "silver_like_dusting", "small_dents_in_nails", 
    "inflammatory_nails", "blister", "red_sore_around_nose", "yellow_crust_ooze"
]


selected_symptoms = st.multiselect("Select the symptoms you are experiencing:", all_symptoms)

def prepare_input(selected_symptoms, all_symptoms):
    input_features = [0] * len(all_symptoms)
    for symptom in selected_symptoms:
        if symptom in all_symptoms:
            index = all_symptoms.index(symptom)
            input_features[index] = 1
    return pd.DataFrame([input_features], columns=all_symptoms)

if st.button("Predict"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        input_df = prepare_input(selected_symptoms, all_symptoms)
        
        prediction = model.predict(input_df)[0]
        
        disease_mapping = {
            0: "Dengue",
            1: "Alcoholic hepatitis",
            2: "(vertigo) Paroymsal Positional Vertigo",
            3: "Diabetes",
            4: "Hyperthyroidism",
            5: "Paralysis (brain hemorrhage)",
            6: "Urinary tract infection",
            7: "Chicken pox",
            8: "Allergy",
            9: "Migraine",
            10: "Hepatitis A",
            11: "Osteoarthritis",
            12: "Cervical spondylosis",
            13: "Common Cold",
            14: "Jaundice",
            15: "Tuberculosis",
            16: "Fungal infection",
            17: "AIDS",
            18: "Peptic ulcer disease",
            19: "Psoriasis",
            20: "Malaria",
            21: "Hypertension",
            22: "Hepatitis C",
            23: "Acne",
            24: "Heart attack",
            25: "Hypoglycemia",
            26: "Impetigo",
            27: "Typhoid",
            28: "Bronchial Asthma",
            29: "Arthritis",
            30: "GERD",
            31: "Hepatitis E",
            32: "Hepatitis D",
            33: "Gastroenteritis",
            34: "Hepatitis B",
            35: "Pneumonia",
            36: "Dimorphic hemorrhoids (piles)",
            37: "Chronic cholestasis",
            38: "Drug Reaction",
            39: "Varicose veins",
            40: "Hypothyroidism"
        }

        predicted_disease = disease_mapping.get(prediction, "Unknown Disease")
        
        st.success(f"Predicted Disease: {predicted_disease}")
        
        st.header("Prescription")

        with st.container():
            st.markdown("### 🥗 Diet Recommendations")
            diet = diets_df.loc[diets_df['Disease'] == predicted_disease, 'Diet']
            if not diet.empty:
                diet_list = diet.iloc[0].strip("[]").replace("'", "").split(", ")
                for item in diet_list:
                    st.write(f"- {item}")
            else:
                st.warning("No diet recommendations available.")

            st.markdown("---")

            with st.container():
                st.markdown("### 💊 Medications")
                medication = medications_df.loc[medications_df['Disease'] == predicted_disease, 'Medication']
                if not medication.empty:
                    medications_list = medication.iloc[0].strip("[]").replace("'", "").split(", ")
                    for med in medications_list:
                        st.write(f"- {med}")
                else:
                    st.warning("No medication recommendations available.")

                st.markdown("---")

        with st.container():
            st.markdown("### ⚠️ Precautions")
            precautions = precautions_df.loc[
                precautions_df['Disease'] == predicted_disease, 
                ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']
            ]
            if not precautions.empty:
                for precaution in precautions.iloc[0].dropna():
                    st.write(f"- {precaution}")
            else:
                st.warning("No precaution recommendations available.")

            st.markdown("---")


        with st.container():
            st.markdown("### 🏋️‍♀️ Workout Suggestions")
            workout = workouts_df.loc[workouts_df['disease'] == predicted_disease, 'workout']
            if not workout.empty:
                st.write(workout.iloc[0])
            else:
                st.warning("No workout suggestions available.")
