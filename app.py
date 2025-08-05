import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ---------------------- Signal Processing Functions ----------------------

# Generate Walsh Codes (Hadamard Matrix)
def generate_walsh_codes(n):
    def hadamard(n):
        if n == 1:
            return np.array([[1]])
        H = hadamard(n // 2)
        return np.vstack((np.hstack((H, H)), np.hstack((H, -H))))
    assert (n & (n - 1) == 0), "Walsh matrix size must be power of 2"
    return hadamard(n)

# Spread a user's data using its code
def spread_data(data, code):
    spread = []
    for bit in data:
        spread.append(code * (1 if bit == 1 else -1))
    return np.concatenate(spread)

# Add AWGN Noise
def add_awgn(signal, snr_db):
    snr = 10**(snr_db / 10)
    power = np.mean(signal**2)
    noise_power = power / snr
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

# Matched Filter for Detection
def matched_filter(received, code, bit_length):
    despread = []
    for i in range(bit_length):
        segment = received[i * len(code):(i + 1) * len(code)]
        correlation = np.dot(segment, code)
        despread.append(1 if correlation > 0 else 0)
    return np.array(despread)

# Bit Error Rate Calculation
def calculate_ber(original, recovered):
    return np.sum(original != recovered) / len(original)

# Cross-correlation between codes
def cross_correlation_matrix(codes):
    size = len(codes)
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            matrix[i, j] = np.dot(codes[i], codes[j]) / len(codes[i])
    return matrix

# ---------------------- Streamlit UI ----------------------

st.set_page_config(page_title="Advanced CDMA Signal Simulator", layout="wide")
st.title("📡 Advanced CDMA Signal Detection Simulator")
st.markdown("### Built from scratch using signal processing fundamentals (No datasets, No ML models)")

# Sidebar Controls
st.sidebar.header("🔧 Controls")
num_users = st.sidebar.slider("Number of Users", 2, 4, 2)
bit_length = st.sidebar.slider("Bits per User", 1, 4, 2)
snr_db = st.sidebar.slider("SNR (dB)", 0, 30, 10)
manual_input = st.sidebar.checkbox("Enter Bits Manually")

# Generate Walsh Codes
walsh = generate_walsh_codes(4)

# Input Section
st.subheader("📶 User Binary Data")
user_data = {}
for i in range(num_users):
    user = f"User {i+1}"
    if manual_input:
        bits_input = st.text_input(f"{user} Bits (comma-separated 0s and 1s)", value="1,0")
        bits = list(map(int, bits_input.split(',')))
    else:
        bits = list(np.random.randint(0, 2, bit_length))
    if len(bits) != bit_length:
        st.error(f"{user} must have exactly {bit_length} bits")
    user_data[user] = np.array(bits)

# Spread and Combine Signals
spread_signals = {}
for i, (user, bits) in enumerate(user_data.items()):
    spread_signals[user] = spread_data(bits, walsh[i])

composite_signal = np.sum(list(spread_signals.values()), axis=0)
received_signal = add_awgn(composite_signal, snr_db)

# Layout: Signal Visualization
st.subheader("📈 Composite Received Signal with Noise")
fig1, ax1 = plt.subplots()
ax1.plot(received_signal, color='blue')
ax1.set_title("Noisy Composite CDMA Signal")
ax1.set_xlabel("Sample Index")
ax1.set_ylabel("Amplitude")
ax1.grid(True)
st.pyplot(fig1)

# Detection Results
st.subheader("🎯 Detection and BER per User")
cols = st.columns(num_users)
for idx, (user, bits) in enumerate(user_data.items()):
    recovered = matched_filter(received_signal, walsh[idx], len(bits))
    ber = calculate_ber(bits, recovered)
    with cols[idx]:
        st.markdown(f"#### {user}")
        st.write(f"Sent Bits: {bits.tolist()}")
        st.write(f"Recovered: {recovered.tolist()}")
        st.write(f"BER: `{ber:.2f}`")

# Cross-Correlation Matrix
st.subheader("🔁 Walsh Code Cross-Correlation Matrix")
corr_matrix = cross_correlation_matrix([walsh[i] for i in range(num_users)])
fig2, ax2 = plt.subplots()
cax = ax2.matshow(corr_matrix, cmap='coolwarm')
plt.title("Cross-Correlation Between Walsh Codes")
fig2.colorbar(cax)
st.pyplot(fig2)

# Footer
st.info("🔬 Project developed using only core Digital Communication logic (No AI, No Datasets, No Frameworks). Built for academic insight and real-time demo.")
