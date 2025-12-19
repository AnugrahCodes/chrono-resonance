import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import altair as alt
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import adfuller
import pywt
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import umap
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import base64

import plotly.io as pio

# Dark mode optimized Plotly theme
pio.templates.default = "plotly_dark"


st.set_page_config(
    page_title="Financial Time Resonance Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------- Utility Functions -------------------

@st.cache_data
def load_data(uploaded_file):
    """Load and preprocess financial data from CSV file"""
    if uploaded_file is None:
        st.warning("Please upload a file to proceed.")
        return None

    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df = df.fillna(method='ffill')
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
    df = df.dropna()
    return df


def calculate_features(data):
    """Calculate additional features for resonance analysis"""
    features = pd.DataFrame(index=data.index)
    
    # Price-based features
    features['close_norm'] = normalize_series(data['Close'])
    features['returns'] = data['Returns']
    features['log_returns'] = np.log1p(data['Returns'])
    features['volatility'] = data['Volatility']
    features['price_range'] = data['Price_Range']
    
    # Moving averages and trends
    features['sma20_close_ratio'] = data['Close'] / data['SMA_20']
    features['sma_ratio'] = data['SMA_20'] / data['SMA_50']
    
    # Volume-based features
    features['volume_norm'] = normalize_series(data['Volume'])
    features['volume_change'] = data['Volume_Change']
    
    # Momentum indicators
    features['momentum_5d'] = data['Close'].pct_change(5)
    features['momentum_10d'] = data['Close'].pct_change(10)
    
    return features

def normalize_series(series):
    """Min-max normalize a series to 0-1 range"""
    return (series - series.min()) / (series.max() - series.min())

def segment_data(data, window_size):
    """Segment the data into windows of specific size"""
    segments = []
    for i in range(0, len(data) - window_size + 1):
        segments.append(data.iloc[i:i+window_size])
    return segments

def calculate_similarity(segment1, segment2, method='pearson'):
    """Calculate similarity between two segments using specified method"""
    if method == 'pearson':
        correlation, _ = pearsonr(segment1, segment2)
        return abs(correlation)  # Use absolute correlation as similarity measure
    elif method == 'dtw':
        distance, _ = fastdtw(segment1, segment2, dist=euclidean)
        # Convert distance to similarity (inverse relationship)
        return 1 / (1 + distance)
    elif method == 'euclidean':
        distance = np.sqrt(np.sum((segment1 - segment2) ** 2))
        return 1 / (1 + distance)
    else:
        return 0

def compute_wavelet_transform(data, wavelet='db4', level=5):
    """Compute wavelet transform for multi-scale analysis"""
    coeffs = pywt.wavedec(data, wavelet, level=level)
    return coeffs

def generate_decompositions(data, column='close_norm'):
    """Generate multi-timeframe decompositions of the data"""
    decompositions = {}
    
    # Daily patterns
    decompositions['daily'] = data[column].values
    
    # Weekly pattern (resample to weekly)
    weekly = data[column].resample('W').mean()
    decompositions['weekly'] = weekly.values
    
    # Monthly pattern
    monthly = data[column].resample('M').mean()
    decompositions['monthly'] = monthly.values
    
    # Quarterly pattern
    quarterly = data[column].resample('Q').mean()
    decompositions['quarterly'] = quarterly.values
    
    # Wavelet decomposition for fractal patterns
    wavelet = compute_wavelet_transform(data[column].values)
    for i, coeff in enumerate(wavelet):
        decompositions[f'wavelet_level_{i}'] = coeff
    
    return decompositions

def find_resonances(data, current_window, window_size, n_results=5):
    """
    Find historical periods that resonate with the current window
    based on multiple features and timeframes
    """
    features = calculate_features(data)
    
    # Get current window features
    if len(data) < window_size:
        return None, None
    
    current_features = features.iloc[-window_size:]
    
    # Segment historical data
    historical_segments = []
    dates = []
    
    for i in range(0, len(features) - window_size - window_size):  # Avoid overlap with current window
        segment = features.iloc[i:i+window_size]
        historical_segments.append(segment)
        dates.append(segment.index[0])
    
    # Calculate resonance scores
    similarity_scores = []
    
    # Features to compare
    feature_columns = ['close_norm', 'returns', 'volatility', 'volume_norm', 'price_range']
    
    for segment in historical_segments:
        segment_scores = []
        
        for feature in feature_columns:
            # Calculate similarity for this feature
            sim = calculate_similarity(
                current_features[feature].values, 
                segment[feature].values
            )
            segment_scores.append(sim)
        
        # Average score across all features
        avg_score = np.mean(segment_scores)
        similarity_scores.append(avg_score)
    
    # Get top resonance periods
    results_df = pd.DataFrame({
        'Start_Date': dates,
        'Resonance_Score': similarity_scores
    })
    
    # Sort by resonance score in descending order
    results_df = results_df.sort_values('Resonance_Score', ascending=False)
    
    # Get top n results
    top_resonances = results_df.head(n_results)
    
    # Extract the actual data segments for the top resonances
    top_segments = []
    for date in top_resonances['Start_Date']:
        idx = data.index.get_indexer([date])[0]
        segment = data.iloc[idx:idx+window_size]
        top_segments.append(segment)
    
    return top_resonances, top_segments

def calculate_future_trajectories(data, resonances, top_segments, window_size, forecast_horizon=30):
    """Calculate potential future trajectories based on historical resonances"""
    trajectories = []
    
    # Calculate the average trajectory
    avg_trajectory = np.zeros(forecast_horizon)
    
    for i, segment in enumerate(top_segments):
        # If we have data after this segment, use it as a potential future trajectory
        start_idx = data.index.get_indexer([segment.index[0]])[0]
        
        if start_idx + window_size + forecast_horizon <= len(data):
            # Extract the future trajectory after this segment
            future_segment = data.iloc[start_idx+window_size:start_idx+window_size+forecast_horizon]
            
            # Normalize the future trajectory to start at the current price
            norm_factor = data['Close'].iloc[-1] / future_segment['Close'].iloc[0]
            future_trajectory = future_segment['Close'].values * norm_factor
            
            # Assign a weight based on the resonance score
            weight = resonances['Resonance_Score'].iloc[i]
            
            trajectories.append({
                'trajectory': future_trajectory,
                'start_date': segment.index[0],
                'resonance_score': weight
            })
            
            # Add weighted contribution to average trajectory
            avg_trajectory += future_trajectory * weight
    
    # Normalize the average trajectory
    if trajectories:
        total_weight = resonances['Resonance_Score'].sum()
        avg_trajectory /= total_weight
    
    return trajectories, avg_trajectory

def visualize_resonances_3d(current_data, historical_segments, scores):
    """Create 3D visualization of resonance relationships"""
    # Extract features for dimensionality reduction
    features_list = []
    
    # Add current data
    current_features = np.concatenate([
        normalize_series(current_data['Close']).values,
        current_data['Volatility'].values,
        normalize_series(current_data['Volume']).values
    ])
    features_list.append(current_features)
    
    # Add historical segments
    for segment in historical_segments:
        segment_features = np.concatenate([
            normalize_series(segment['Close']).values,
            segment['Volatility'].values,
            normalize_series(segment['Volume']).values
        ])
        features_list.append(segment_features)
    
    # Perform dimensionality reduction with UMAP
    reducer = umap.UMAP(n_components=3, random_state=42)
    embedding = reducer.fit_transform(features_list)
    
    # Create 3D plot
    fig = go.Figure()
    
    # Plot historical segments
    for i in range(1, len(embedding)):
        fig.add_trace(go.Scatter3d(
            x=[embedding[i, 0]],
            y=[embedding[i, 1]],
            z=[embedding[i, 2]],
            mode='markers',
            marker=dict(
                size=10,
                color=scores[i-1],
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title='Resonance Score')
            ),
            text=f"Start Date: {historical_segments[i-1].index[0].strftime('%Y-%m-%d')}<br>Score: {scores[i-1]:.4f}",
            hoverinfo='text',
            name=f"Historical {i}"
        ))
    
    # Plot current data point with larger marker
    fig.add_trace(go.Scatter3d(
        x=[embedding[0, 0]],
        y=[embedding[0, 1]],
        z=[embedding[0, 2]],
        mode='markers',
        marker=dict(
            size=15,
            color='red',
            symbol='diamond'
        ),
        text="Current Market Position",
        hoverinfo='text',
        name="Current"
    ))
    
    # Add lines connecting current position to top resonances
    for i in range(1, min(4, len(embedding))):
        fig.add_trace(go.Scatter3d(
            x=[embedding[0, 0], embedding[i, 0]],
            y=[embedding[0, 1], embedding[i, 1]],
            z=[embedding[0, 2], embedding[i, 2]],
            mode='lines',
            line=dict(
                color='rgba(100, 100, 100, 0.4)',
                width=2
            ),
            hoverinfo='none',
            showlegend=False
        ))
    
    fig.update_layout(
        title="3D Resonance Field Map",
        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Dimension 3"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def compare_patterns(current_segment, historical_segment):
    """Create comparison visualization between current and historical patterns"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Price Comparison", "Volume Comparison"),
        vertical_spacing=0.15
    )
    
    # Normalize both segments to start at 100 for better comparison
    current_norm = 100 * current_segment['Close'] / current_segment['Close'].iloc[0]
    historical_norm = 100 * historical_segment['Close'] / historical_segment['Close'].iloc[0]
    
    # Create date ranges for x-axis
    current_dates = np.arange(len(current_segment))
    historical_dates = np.arange(len(historical_segment))
    
    # Price comparison
    fig.add_trace(
        go.Scatter(
            x=current_dates,
            y=current_norm,
            mode='lines',
            name='Current',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=historical_dates,
            y=historical_norm,
            mode='lines',
            name='Historical',
            line=dict(color='red')
        ),
        row=1, col=1
    )
    
    # Volume comparison
    current_vol_norm = current_segment['Volume'] / current_segment['Volume'].mean()
    historical_vol_norm = historical_segment['Volume'] / historical_segment['Volume'].mean()
    
    fig.add_trace(
        go.Bar(
            x=current_dates,
            y=current_vol_norm,
            name='Current Vol',
            marker_color='rgba(0, 0, 255, 0.5)'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(
            x=historical_dates,
            y=historical_vol_norm,
            name='Historical Vol',
            marker_color='rgba(255, 0, 0, 0.5)'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title="Pattern Comparison",
        xaxis_title="Trading Days",
        yaxis_title="Normalized Price (Start=100)",
        xaxis2_title="Trading Days",
        yaxis2_title="Normalized Volume",
        legend_title="Pattern Type",
        height=600
    )
    
    return fig

def visualize_future_trajectories(
    data,
    trajectories,
    avg_trajectory,
    forecast_horizon=30
):
    fig = go.Figure()

    # ---------- Historical data ----------
    historical_days = 60
    hist_prices = data['Close'].iloc[-historical_days:]

    fig.add_trace(go.Scatter(
        x=list(range(-historical_days+1, 1)),
        y=hist_prices.values,
        mode='lines',
        name='Historical Price',
        line=dict(color='#AAAAAA', width=2)
    ))

    # ---------- Future trajectories ----------
    future_matrix = []

    for traj in trajectories:
        future_matrix.append(traj['trajectory'])

        fig.add_trace(go.Scatter(
            x=list(range(1, forecast_horizon + 1)),
            y=traj['trajectory'],
            mode='lines',
            line=dict(color='rgba(100,150,255,0.25)'),
            showlegend=False
        ))

    future_matrix = np.array(future_matrix)

    # ---------- Confidence bands ----------
    mean_path = np.mean(future_matrix, axis=0)
    upper_band = np.percentile(future_matrix, 75, axis=0)
    lower_band = np.percentile(future_matrix, 25, axis=0)

    fig.add_trace(go.Scatter(
        x=list(range(1, forecast_horizon + 1)),
        y=upper_band,
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=list(range(1, forecast_horizon + 1)),
        y=lower_band,
        fill='tonexty',
        fillcolor='rgba(0,200,150,0.2)',
        line=dict(width=0),
        name='Confidence Band (25–75%)'
    ))

    # ---------- Mean projection ----------
    fig.add_trace(go.Scatter(
        x=list(range(1, forecast_horizon + 1)),
        y=mean_path,
        mode='lines',
        name='Expected Path',
        line=dict(color='#00E5FF', width=3)
    ))

    # ---------- Regime detection ----------
    baseline = data['Close'].iloc[-1]
    expected_return = mean_path[-1] / baseline - 1

    if expected_return > 0.05:
        regime = "Bullish Regime 🟢"
        color = "lime"
    elif expected_return < -0.05:
        regime = "Bearish Regime 🔴"
        color = "red"
    else:
        regime = "Neutral Regime 🟡"
        color = "orange"

    fig.add_annotation(
        x=forecast_horizon * 0.6,
        y=max(upper_band),
        text=f"<b>{regime}</b><br>Expected Return: {expected_return:.2%}",
        showarrow=False,
        font=dict(color=color, size=14),
        bgcolor="rgba(0,0,0,0.6)"
    )

    # ---------- Layout ----------
    fig.update_layout(
        title="Future Price Projection with Confidence Envelope",
        xaxis_title="Days Ahead",
        yaxis_title="Price",
        hovermode="x unified",
        height=550
    )

    return fig


def create_temporal_harmony_dashboard(data, resonances, window_size):
    """Create visualizations showing resonance strength across different timeframes"""
    if resonances is None or resonances.empty:
        return None
    
    # Create datetime bins for visualization
    resonances['Year'] = resonances['Start_Date'].dt.year
    resonances['Month'] = resonances['Start_Date'].dt.month
    resonances['Quarter'] = resonances['Start_Date'].dt.quarter
    
    # Create heatmap data for years and months
    year_counts = resonances.groupby('Year')['Resonance_Score'].mean().reset_index()
    month_counts = resonances.groupby('Month')['Resonance_Score'].mean().reset_index()
    
    # Year resonance
    year_fig = px.bar(
        year_counts,
        x='Year',
        y='Resonance_Score',
        title='Resonance Strength by Year',
        labels={'Resonance_Score': 'Average Resonance'},
        color='Resonance_Score',
        color_continuous_scale='Viridis'
    )
    
    # Month resonance (cyclic pattern)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_counts['Month_Name'] = month_counts['Month'].apply(lambda x: month_names[x-1])
    
    month_fig = px.bar(
        month_counts,
        x='Month',
        y='Resonance_Score',
        title='Seasonal Resonance Strength',
        labels={'Resonance_Score': 'Average Resonance', 'Month': 'Month'},
        color='Resonance_Score',
        color_continuous_scale='Viridis'
    )
    month_fig.update_layout(xaxis = dict(
        tickmode = 'array',
        tickvals = list(range(1, 13)),
        ticktext = month_names
    ))
    
    # Create timeseries of all resonances
    timeseries_fig = px.scatter(
        resonances,
        x='Start_Date',
        y='Resonance_Score',
        color='Resonance_Score',
        size='Resonance_Score',
        title='Historical Resonance Distribution',
        labels={'Resonance_Score': 'Resonance Strength', 'Start_Date': 'Date'},
        color_continuous_scale='Viridis'
    )
    timeseries_fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    
    return year_fig, month_fig, timeseries_fig

def calculate_fractal_dimension(data, window_sizes):
    """Calculate fractal dimension using box counting method"""
    dimensions = []
    
    normalized_data = normalize_series(data)
    
    for size in window_sizes:
        # Count number of boxes needed to cover the data
        n_boxes = len(data) // size
        count = 0
        
        for i in range(n_boxes):
            start_idx = i * size
            end_idx = start_idx + size
            
            if end_idx <= len(data):
                # Check if this box contains part of the curve
                segment = normalized_data[start_idx:end_idx]
                if segment.max() - segment.min() > 0:
                    count += 1
        
        dimensions.append((np.log(1/size), np.log(count)))
    
    # Calculate slope of log-log plot
    if dimensions:
        x, y = zip(*dimensions)
        x, y = np.array(x), np.array(y)
        
        # Remove any inf or NaN values
        valid_indices = ~np.isnan(x) & ~np.isnan(y) & ~np.isinf(x) & ~np.isinf(y)
        x, y = x[valid_indices], y[valid_indices]
        
        if len(x) > 1:
            slope, _ = np.polyfit(x, y, 1)
            return slope
    
    return None

def calculate_cycle_periods(data, column='Close'):
    """Calculate dominant cycle periods using FFT"""
    # Detrend data
    price = data[column].values
    x = np.arange(len(price))
    
    # Fit linear trend
    slope, intercept = np.polyfit(x, price, 1)
    trend = slope * x + intercept
    
    # Remove trend
    detrended = price - trend
    
    # Apply FFT
    fft_result = np.fft.fft(detrended)
    frequency = np.fft.fftfreq(len(detrended))
    
    # Get positive frequencies only
    positive_freq_idx = np.where(frequency > 0)
    frequency = frequency[positive_freq_idx]
    magnitude = np.abs(fft_result[positive_freq_idx])
    
    # Find peaks in the frequency domain
    peaks, _ = signal.find_peaks(magnitude, height=np.mean(magnitude))
    
    # Calculate period from frequency
    if len(peaks) > 0:
        peak_freqs = frequency[peaks]
        peak_magnitudes = magnitude[peaks]
        
        # Convert frequency to periods (in days)
        periods = np.round(1 / peak_freqs).astype(int)
        
        # Sort by magnitude (importance)
        sorted_idx = np.argsort(peak_magnitudes)[::-1]
        sorted_periods = periods[sorted_idx]
        sorted_magnitudes = peak_magnitudes[sorted_idx]
        
        # Return top periods and their strengths
        return sorted_periods[:5], sorted_magnitudes[:5] / np.sum(peak_magnitudes)
    
    return [], []

# ------------------- Application Layout -------------------

def main():

    # ---------- PAGE HEADER ----------
    st.markdown("""
    <style>
    .big-title {
        font-size: 42px;
        font-weight: 700;
        color: #1f77b4;
    }
    .subtitle {
        font-size: 18px;
        color: #6c757d;
        margin-bottom: 10px;
    }
    .divider {
        height: 4px;
        background: linear-gradient(to right, #1f77b4, #00c9a7);
        border-radius: 5px;
        margin: 10px 0 30px 0;
    }
    </style>

    <div class="big-title">ChronoResonance</div>
    <div class="subtitle">
    AI-Driven Financial Time Resonance Engine for Market Pattern Analysis
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ChronoResonance identifies **temporal resonance patterns** in financial markets —
    situations where current market conditions strongly echo historical periods across
    multiple dimensions of time, volatility, volume, and structure.
    """)

    # ---------- SIDEBAR CONTROL PANEL ----------
    st.sidebar.markdown("## ⚙️ Control Panel")

    uploaded_file = st.sidebar.file_uploader(
        "📁 Upload Financial Data (CSV)",
        type=["csv"]
    )

    if not uploaded_file:
        st.info("Please upload a CSV file with columns: Date, Open, High, Low, Close, Adj Close, Volume")
        st.stop()

    # Load data
    data = load_data(uploaded_file)

    # ---------- DATA STATS ----------
    st.sidebar.markdown("### 📊 Dataset Overview")
    st.sidebar.text(f"Date Range:\n{data.index[0].date()} → {data.index[-1].date()}")
    st.sidebar.text(f"Total Trading Days: {len(data)}")

    # ---------- ANALYSIS PARAMETERS ----------
    with st.sidebar.expander("🧮 Analysis Parameters", expanded=True):
        window_size = st.slider("Analysis Window (Days)", 5, 252, 60)
        n_results = st.slider("Resonance Patterns", 3, 10, 5)
        forecast_horizon = st.slider("Forecast Horizon (Days)", 5, 90, 30)

    with st.sidebar.expander("🎯 Feature Weighting"):
        price_weight = st.slider("Price Importance", 0.1, 1.0, 0.6)
        volume_weight = st.slider("Volume Importance", 0.0, 1.0, 0.2)
        volatility_weight = st.slider("Volatility Importance", 0.0, 1.0, 0.2)

    # ---------- VALIDATION ----------
    if len(data) < window_size * 2:
        st.error(
            f"Not enough data for analysis. "
            f"Need at least {window_size * 2} days, but got {len(data)}."
        )
        st.stop()

    # ---------- CORE ANALYSIS ----------
    current_window = data.iloc[-window_size:]

    with st.spinner("🔍 Detecting historical resonance patterns..."):
        resonances, historical_segments = find_resonances(
            data, current_window, window_size, n_results
        )

    # ---------- MAIN TABS ----------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Resonance Dashboard",
        "🔍 Pattern Comparison",
        "🔮 Future Trajectories",
        "🧭 Temporal Harmony",
        "🧬 Fractal Analysis"
    ])

    # ================= TAB 1 =================
    with tab1:
        st.markdown("## 📈 Market Resonance Dashboard")

        col1, col2, col3 = st.columns(3)

        last_price = data['Close'].iloc[-1]
        price_change = last_price / data['Close'].iloc[-2] - 1
        volatility = data['Volatility'].iloc[-1]

        col1.metric("Last Price", f"{last_price:.2f}", f"{price_change:.2%}")
        col2.metric("20-Day Volatility", f"{volatility:.2%}")
        col3.metric("Trend Bias", "Bullish" if price_change > 0 else "Bearish")

        st.markdown("### 📉 Recent Market Trend (60 Days)")
        recent_data = data.iloc[-60:]
        fig = px.line(recent_data, y='Close')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🔗 Top Resonance Patterns")

        if resonances is not None and not resonances.empty:
            for i, row in resonances.iterrows():
                with st.expander(
                    f"Pattern {resonances.index.get_loc(i)+1} | "
                    f"{row['Start_Date'].strftime('%Y-%m-%d')} "
                    f"(Score: {row['Resonance_Score']:.4f})"
                ):
                    segment = historical_segments[resonances.index.get_loc(i)]
                    st.line_chart(segment['Close'])

        st.markdown("### 🌐 3D Resonance Field Map")

        if resonances is not None and not resonances.empty:
            fig_3d = visualize_resonances_3d(
                current_window,
                historical_segments,
                resonances['Resonance_Score'].values
            )
            st.plotly_chart(fig_3d, use_container_width=True)

            st.info("""
            **Interpretation Guide**
            - Red diamond → current market state  
            - Closer points → stronger historical similarity  
            - Lines → strongest resonance connections
            """)

    # ================= TAB 2 =================
    with tab2:
        st.markdown("## 🔍 Pattern Comparison")

        if resonances is not None and not resonances.empty:
            pattern_idx = st.selectbox(
                "Select a historical resonance pattern",
                range(len(resonances)),
                format_func=lambda i:
                f"Pattern {i+1} | {resonances['Start_Date'].iloc[i].strftime('%Y-%m-%d')}"
            )

            fig = compare_patterns(current_window, historical_segments[pattern_idx])
            st.plotly_chart(fig, use_container_width=True)

    # ================= TAB 3 =================
    with tab3:
        st.markdown("## 🔮 Future Trajectory Projection")

        if resonances is not None and not resonances.empty:
            trajectories, avg_trajectory = calculate_future_trajectories(
                data, resonances, historical_segments,
                window_size, forecast_horizon
            )

            if trajectories:
                fig = visualize_future_trajectories(
                    data, trajectories, avg_trajectory, forecast_horizon
                )
                st.plotly_chart(fig, use_container_width=True)

    # ================= TAB 4 =================
    with tab4:
        st.markdown("## 🧭 Temporal Harmony Analysis")

        if resonances is not None and not resonances.empty:
            year_fig, month_fig, ts_fig = create_temporal_harmony_dashboard(
                data, resonances, window_size
            )
            st.plotly_chart(year_fig, use_container_width=True)
            st.plotly_chart(month_fig, use_container_width=True)
            st.plotly_chart(ts_fig, use_container_width=True)

    # ================= TAB 5 =================
    with tab5:
        st.markdown("## 🧬 Fractal & Wavelet Analysis")

        fractal_dim = calculate_fractal_dimension(
            data['Close'], [5, 10, 20, 40, 80]
        )

        if fractal_dim:
            st.metric("Fractal Dimension", f"{fractal_dim:.4f}")

        st.info("""
        **Fractal Interpretation**
        - < 1.3 → Trending market  
        - 1.3 – 1.6 → Mixed regime  
        - > 1.6 → Chaotic behavior
        """)

    # ---------- FOOTER ----------
    st.markdown("---")
    st.caption("ChronoResonance • Market Memory through Time • Built with Streamlit & Python")


if __name__ == "__main__":
    main()
