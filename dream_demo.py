import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ks_2samp
import tensorflow as tf
from tensorflow.keras import layers

# Set Streamlit Page Config
st.set_page_config(page_title="DREAM Project Workflow", layout="wide")

# Introduction Section
st.title("DREAM: Data Realization through Efficient Adaptive Modeling")
st.write("""
This interactive dashboard demonstrates the technical workflow of generating 
statistically sound and contextually relevant synthetic data using **GMM**, 
**VAE**, **GAN**, and a **Feedback Loop**. 
""")

# Step 1: Input Data
st.header("Step 1: Input Data")
st.write("Upload a dataset or define parameters to generate initial synthetic data.")

# File Upload or Data Generation
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", data.head())
else:
    st.subheader("Or Generate Data")
    samples = st.slider("Number of Samples", 50, 1000, 200)
    mean = st.number_input("Mean", value=0.0)
    std_dev = st.number_input("Standard Deviation", value=1.0)
    data = pd.DataFrame(np.random.normal(mean, std_dev, size=(samples, 2)), columns=["X1", "X2"])
    st.write("Generated Data Preview:", data.head())

# Normalize Data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)

# Visualize Input Data
st.write("Data Visualization:")
st.scatter_chart(data)

# Step 2: Gaussian Mixture Model (GMM)
st.header("Step 2: Gaussian Mixture Model (GMM)")
st.write("""
GMM identifies patterns in the input data by modeling it as a combination of Gaussian distributions.
""")
n_components = st.slider("Number of Gaussian Components", 1, 5, 2)

# Fit GMM
gmm = GaussianMixture(n_components=n_components, random_state=42)
gmm.fit(normalized_data)
synthetic_gmm, _ = gmm.sample(len(data))

# Visualize GMM Output
fig, ax = plt.subplots()
ax.scatter(normalized_data[:, 0], normalized_data[:, 1], label="Original Data", alpha=0.5)
ax.scatter(synthetic_gmm[:, 0], synthetic_gmm[:, 1], label="GMM Synthetic Data", alpha=0.5, color='orange')
ax.legend()
st.pyplot(fig)

# Step 3: Variational Autoencoder (VAE)
st.header("Step 3: Variational Autoencoder (VAE)")
st.write("""
VAE refines the GMM-generated data by encoding it into a latent space and reconstructing it.
""")
latent_dim = st.slider("Latent Space Dimensions", 2, 10, 2)

# VAE Model
input_dim = synthetic_gmm.shape[1]
encoder_inputs = tf.keras.Input(shape=(input_dim,))
x = layers.Dense(64, activation="relu")(encoder_inputs)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])
encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

decoder_inputs = tf.keras.Input(shape=(latent_dim,))
x = layers.Dense(64, activation="relu")(decoder_inputs)
decoder_outputs = layers.Dense(input_dim, activation="sigmoid")(x)
decoder = tf.keras.Model(decoder_inputs, decoder_outputs, name="decoder")

vae = tf.keras.Model(encoder_inputs, decoder(encoder(encoder_inputs)[2]), name="vae")
vae.compile(optimizer="adam", loss="mse")
vae.fit(synthetic_gmm, synthetic_gmm, epochs=10, batch_size=32, verbose=0)

# Generate VAE Output
vae_output = vae.predict(synthetic_gmm)

# Visualize VAE Output
fig, ax = plt.subplots()
ax.scatter(synthetic_gmm[:, 0], synthetic_gmm[:, 1], label="GMM Synthetic Data", alpha=0.5)
ax.scatter(vae_output[:, 0], vae_output[:, 1], label="VAE Refined Data", alpha=0.5, color='green')
ax.legend()
st.pyplot(fig)

# Step 4: Generative Adversarial Network (GAN)
st.header("Step 4: Generative Adversarial Network (GAN)")
st.write("""
GAN enhances the realism of the synthetic data through adversarial training.
""")
# Placeholder for GAN (could implement later with detailed training progress visualization)

st.write("🚧 GAN visualization under construction.")

# Step 5: Feedback Loop
st.header("Step 5: Feedback Loop")
st.write("""
Statistical tests validate the synthetic data and provide feedback for refinement.
""")

# Kolmogorov-Smirnov Test
ks_stat, ks_p = ks_2samp(normalized_data.flatten(), vae_output.flatten())
st.write(f"Kolmogorov-Smirnov Test Statistic: {ks_stat:.4f}, P-value: {ks_p:.4f}")

# Histograms for Visual Comparison
fig, ax = plt.subplots()
ax.hist(normalized_data.flatten(), bins=30, alpha=0.5, label="Original Data")
ax.hist(vae_output.flatten(), bins=30, alpha=0.5, label="Refined Data")
ax.legend()
st.pyplot(fig)

# Step 6: Final Output
st.header("Step 6: Final Output")
st.write("Download the final synthetic dataset.")
final_data = pd.DataFrame(vae_output, columns=["X1", "X2"])
csv = final_data.to_csv(index=False)
st.download_button("Download Final Dataset", csv, "synthetic_data.csv", "text/csv")
