#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Application de prédiction des cours boursiers Johnson & Johnson (JNJ)
Modèles utilisés : LSTM (PyTorch) et NeuralProphet
Auteur : Adaptation avec PyTorch
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# NeuralProphet
from neuralprophet import NeuralProphet

# Configuration
torch.manual_seed(42)
np.random.seed(42)

st.set_page_config(
    page_title="JNJ Stock Predictor - PyTorch", 
    page_icon="🔥", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Palette de couleurs MODIFIÉE POUR MODE SOMBRE
COLORS = {
    'primary': '#A78BFA',      # Violet clair pour ressortir sur le noir
    'secondary': '#C084FC',
    'tertiary': '#8B5CF6',
    'gradient_start': '#7C3AED',
    'gradient_end': '#4C1D95',
    'background': '#0F172A',   # Bleu/Noir profond
    'card_bg': '#1E293B',      # Fond des cartes
    'text': '#F8FAFC',         # Texte clair
    'success': '#10B981',
    'warning': '#F59E0B',
    'danger': '#EF4444'
}

# CSS personnalisé - ADAPTATION DARK MODE
def apply_custom_css():
    st.markdown(f"""
    <style>
        /* Fond global de l'application */
        .stApp {{
            background: {COLORS['background']};
            color: {COLORS['text']};
        }}
        
        /* Sidebar sombre */
        section[data-testid="stSidebar"] {{
            background-color: #020617 !important;
        }}
        
        @keyframes fadeIn {{
            0% {{ opacity: 0; transform: translateY(20px); }}
            100% {{ opacity: 1; transform: translateY(0); }}
        }}
        
        @keyframes pulse {{
            0% {{ box-shadow: 0 0 0 0 rgba(167, 139, 250, 0.4); }}
            70% {{ box-shadow: 0 0 0 10px rgba(167, 139, 250, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(167, 139, 250, 0); }}
        }}
        
        .gradient-card {{
            background: {COLORS['card_bg']};
            border-radius: 20px;
            padding: 25px;
            margin: 15px 0;
            border-left: 5px solid {COLORS['primary']};
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
            animation: fadeIn 0.8s ease-out;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            color: {COLORS['text']};
        }}
        
        .gradient-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
        }}
        
        .pulse-card {{
            animation: pulse 2s infinite;
        }}
        
        h1, h2, h3 {{
            background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700 !important;
        }}
        
        .stButton > button {{
            background: linear-gradient(135deg, {COLORS['gradient_start']}, {COLORS['gradient_end']});
            color: white;
            border: none;
            border-radius: 12px;
            padding: 12px 28px;
            font-weight: 600;
            font-size: 16px;
            transition: all 0.3s ease;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
            width: 100%;
        }}
        
        .stButton > button:hover {{
            transform: scale(1.02);
            box-shadow: 0 8px 25px rgba(124, 58, 237, 0.4);
        }}
        
        .metric-container {{
            background: {COLORS['card_bg']};
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            border-bottom: 3px solid {COLORS['primary']};
        }}
        
        .metric-value {{
            font-size: 28px;
            font-weight: 700;
            color: {COLORS['primary']};
        }}
        
        .metric-label {{
            font-size: 14px;
            color: #94A3B8;
        }}
        
        .stProgress > div > div > div {{
            background: linear-gradient(90deg, {COLORS['gradient_start']}, {COLORS['gradient_end']});
        }}
        
        .pytorch-badge {{
            background: linear-gradient(135deg, #EE4C2C, #FF6F61);
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 10px;
        }}
        
        .info-box {{
            background: {COLORS['card_bg']};
            border-radius: 12px;
            padding: 15px;
            border-left: 4px solid {COLORS['primary']};
            margin: 10px 0;
            color: {COLORS['text']};
        }}
        
        /* Timeline steps Dark Mode */
        .step {{
            display: flex;
            align-items: center;
            margin: 20px 0;
            padding: 15px;
            background: {COLORS['card_bg']};
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            color: {COLORS['text']};
        }}
        
        .step-number {{
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, {COLORS['gradient_start']}, {COLORS['gradient_end']});
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            margin-right: 15px;
        }}
    </style>
    """, unsafe_allow_html=True)

# ============================================
# MODÈLE LSTM AVEC PYTORCH
# ============================================

class LSTMPredictor(nn.Module):
    """Modèle LSTM avec PyTorch"""
    def __init__(self, input_size=1, hidden_size=100, num_layers=3, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Couches LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        
        # Couches fully connected
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 50),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(25, 1)
        )
        
    def forward(self, x):
        # Initialisation des états cachés
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagation LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Prendre la dernière sortie
        out = out[:, -1, :]
        
        # Fully connected layers
        out = self.fc(out)
        
        return out

class PyTorchTrainer:
    """Classe pour l'entraînement du modèle PyTorch"""
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = None
        self.history = {'train_loss': [], 'val_loss': []}
        
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, patience=15):
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Barres de progression
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Sauvegarder l'historique
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            
            # Mise à jour du learning rate
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                status_text.text(f"Early stopping à l'epoch {epoch+1}")
                break
            
            # Mise à jour de la progression
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            
            if (epoch + 1) % 10 == 0:
                status_text.text(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Restaurer le meilleur modèle
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        progress_bar.empty()
        status_text.empty()
        
        return self.history

# ============================================
# FONCTIONS PRINCIPALES
# ============================================

@st.cache_data(ttl=3600)
def download_stock_data(ticker='JNJ', period='4y'):
    """Télécharge les données boursières depuis Yahoo Finance"""
    try:
        with st.spinner('📊 Téléchargement des données JNJ...'):
            data = yf.download(ticker, period=period, progress=False)
            
            if data.empty:
                st.error("❌ Aucune donnée téléchargée")
                return None, None
            
            # Métadonnées de l'entreprise
            stock = yf.Ticker(ticker)
            info = stock.info
            
            company_info = {
                'name': info.get('longName', 'Johnson & Johnson'),
                'sector': info.get('sector', 'Healthcare'),
                'industry': info.get('industry', 'Pharmaceuticals'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'volume': info.get('volume', 0),
                'avg_volume': info.get('averageVolume', 0)
            }
            
            return data, company_info
    except Exception as e:
        st.error(f"Erreur lors du téléchargement: {str(e)}")
        return None, None

def prepare_data_pytorch(data, look_back=60, batch_size=32):
    """Prépare les données pour PyTorch"""
    df = data[['Close']].copy()
    
    # Normalisation
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    
    # Création des séquences
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Split train/test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Conversion en tenseurs PyTorch
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Création des DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'X_train': X_train_tensor,
        'y_train': y_train_tensor,
        'X_test': X_test_tensor,
        'y_test': y_test_tensor,
        'scaler': scaler,
        'scaled_data': scaled_data,
        'look_back': look_back
    }

def prepare_neuralprophet_data(data):
    """Prépare les données pour NeuralProphet"""
    df_prophet = data[['Close']].reset_index()
    df_prophet.columns = ['ds', 'y']
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    return df_prophet

def train_neuralprophet(df_prophet):
    """Entraîne un modèle NeuralProphet"""
    model = NeuralProphet(
        growth='linear',
        changepoints_range=0.8,
        n_changepoints=25,
        yearly_seasonality='auto',
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        epochs=100,
        learning_rate=0.01,
        batch_size=64
    )
    
    with st.spinner("Entraînement NeuralProphet..."):
        metrics = model.fit(df_prophet, freq='D', validation_df=df_prophet, progress=False)
    
    return model

def predict_future_lstm_pytorch(model, last_sequence, scaler, days=21, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Prédictions futures avec LSTM PyTorch"""
    model.eval()
    predictions = []
    current_seq = torch.FloatTensor(last_sequence).reshape(1, -1, 1).to(device)
    
    progress_bar = st.progress(0)
    
    with torch.no_grad():
        for i in range(days):
            # Prédiction
            next_pred = model(current_seq)
            predictions.append(next_pred.cpu().item())
            
            # Mise à jour de la séquence
            current_seq = torch.roll(current_seq, -1, dims=1)
            current_seq[0, -1, 0] = next_pred
            
            progress_bar.progress((i + 1) / days)
    
    progress_bar.empty()
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

def plot_interactive_predictions(historical, lstm_pred, prophet_pred, dates):
    """Crée un graphique interactif avec Plotly - ADAPTATION MODE SOMBRE"""
    fig = go.Figure()
    
    # Données historiques
    fig.add_trace(go.Scatter(
        x=historical.index[-252:],
        y=historical['Close'][-252:],
        mode='lines',
        name='Historique',
        line=dict(color=COLORS['primary'], width=2),
        hovertemplate='Date: %{x}<br>Prix: $%{y:.2f}<extra></extra>'
    ))
    
    # Prédictions LSTM
    fig.add_trace(go.Scatter(
        x=dates,
        y=lstm_pred,
        mode='lines+markers',
        name='LSTM (PyTorch)',
        line=dict(color='#FF6B6B', width=2, dash='dash'),
        marker=dict(size=8, symbol='circle'),
        hovertemplate='Date: %{x}<br>Prix LSTM: $%{y:.2f}<extra></extra>'
    ))
    
    # Prédictions NeuralProphet
    fig.add_trace(go.Scatter(
        x=dates,
        y=prophet_pred,
        mode='lines+markers',
        name='NeuralProphet',
        line=dict(color='#4ECDC4', width=2, dash='dash'),
        marker=dict(size=8, symbol='square'),
        hovertemplate='Date: %{x}<br>Prix Prophet: $%{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Prédictions JNJ - LSTM (PyTorch) vs NeuralProphet',
        xaxis_title='Date',
        yaxis_title='Prix ($)',
        hovermode='x unified',
        template='plotly_dark', # Template sombre activé
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_xaxes(gridcolor='#334155', gridwidth=0.5)
    fig.update_yaxes(gridcolor='#334155', gridwidth=0.5)
    
    return fig

# ============================================
# APPLICATION PRINCIPALE
# ============================================

def main():
    apply_custom_css()
    
    # Badge PyTorch
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 10px;">
        <span class="pytorch-badge">🔥 PyTorch</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="font-size: 50px;">🏥</h1>
            <h2>JNJ Predictor</h2>
            <p style="color: #94A3B8;">Johnson & Johnson avec PyTorch</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Informations modèle
        st.markdown("""
        <div class="gradient-card">
            <h3 style="margin-top: 0;">🧠 Architecture LSTM</h3>
            <ul>
                <li>3 couches LSTM (100 hidden)</li>
                <li>Dropout 20%</li>
                <li>3 couches Dense</li>
                <li>Optimiseur: Adam</li>
                <li>Loss: MSE</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Paramètres
        st.markdown("### ⚙️ Paramètres")
        prediction_days = st.slider(
            "Jours de prédiction",
            min_value=7,
            max_value=60,
            value=21,
            step=7
        )
        
        look_back = st.slider(
            "Fenêtre temporelle LSTM",
            min_value=30,
            max_value=120,
            value=60,
            step=10
        )
        
        batch_size = st.selectbox(
            "Batch size",
            options=[16, 32, 64, 128],
            index=1
        )
        
        epochs = st.slider(
            "Epochs maximum",
            min_value=50,
            max_value=200,
            value=100,
            step=25
        )
        
        st.markdown("---")
        
        # Device info
        device = 'CUDA' if torch.cuda.is_available() else 'CPU'
        st.info(f"💻 Device: {device}")
        
        st.caption("© 2024 - Application PyTorch")

    # Header principal
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <h1>Johnson & Johnson</h1>
            <h3 style="color: #94A3B8;">Prédiction avec PyTorch LSTM & NeuralProphet</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Timeline des étapes
    st.markdown("""
    <div style="display: flex; justify-content: space-between; margin: 30px 0;">
        <div class="step"><span class="step-number">1</span> Chargement</div>
        <div class="step"><span class="step-number">2</span> Préparation</div>
        <div class="step"><span class="step-number">3</span> Entraînement</div>
        <div class="step"><span class="step-number">4</span> Prédiction</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Étape 1: Téléchargement
    st.markdown("## 📥 Étape 1: Chargement des données")
    
    if st.button("🚀 Télécharger les données JNJ", use_container_width=True):
        data, company_info = download_stock_data('JNJ')
        
        if data is not None:
            st.session_state['data'] = data
            st.session_state['company_info'] = company_info
            
            # Métriques
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{company_info['market_cap']/1e9:.1f}B$</div>
                    <div class="metric-label">Capitalisation</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{company_info['pe_ratio']:.2f}</div>
                    <div class="metric-label">P/E Ratio</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{company_info['dividend_yield']*100:.2f}%</div>
                    <div class="metric-label">Dividende</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{data.shape[0]}</div>
                    <div class="metric-label">Jours de données</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Graphique
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close',
                line=dict(color=COLORS['primary'], width=2),
                fill='tozeroy',
                fillcolor=f"rgba(139, 92, 246, 0.1)"
            ))
            fig.update_layout(
                title="Historique des cours JNJ",
                xaxis_title="Date",
                yaxis_title="Prix ($)",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Étape 2: Préparation
    if 'data' in st.session_state:
        st.markdown("## 🛠️ Étape 2: Préparation des données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧠 Préparer pour LSTM (PyTorch)", use_container_width=True):
                with st.spinner("Préparation des données PyTorch..."):
                    pytorch_data = prepare_data_pytorch(
                        st.session_state['data'], 
                        look_back,
                        batch_size
                    )
                    
                    st.session_state['pytorch_data'] = pytorch_data
                    
                    st.success(f"✅ Données préparées: {pytorch_data['X_train'].shape[0]} séquences")
                    
                    st.markdown(f"""
                    <div class="gradient-card">
                        <h4>📊 PyTorch Datasets</h4>
                        <p>• Train: {pytorch_data['X_train'].shape[0]} samples</p>
                        <p>• Test: {pytorch_data['X_test'].shape[0]} samples</p>
                        <p>• Sequence length: {look_back}</p>
                        <p>• Batch size: {batch_size}</p>
                        <p>• Device: {device}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            if st.button("📈 Préparer pour NeuralProphet", use_container_width=True):
                with st.spinner("Préparation des données NeuralProphet..."):
                    df_prophet = prepare_neuralprophet_data(st.session_state['data'])
                    st.session_state['prophet_data'] = df_prophet
                    
                    st.success(f"✅ Données préparées: {len(df_prophet)} points")
                    
                    st.markdown(f"""
                    <div class="gradient-card">
                        <h4>📊 NeuralProphet</h4>
                        <p>• Période: {df_prophet['ds'].min().date()} → {df_prophet['ds'].max().date()}</p>
                        <p>• Points: {len(df_prophet)}</p>
                        <p>• Fréquence: Journalière</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Étape 3: Entraînement
    if 'pytorch_data' in st.session_state and 'prophet_data' in st.session_state:
        st.markdown("## 🏋️ Étape 3: Entraînement")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 Entraîner LSTM (PyTorch)", use_container_width=True):
                with st.spinner("Création du modèle PyTorch..."):
                    # Initialisation
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    model = LSTMPredictor(
                        input_size=1,
                        hidden_size=100,
                        num_layers=3,
                        dropout=0.2
                    )
                    
                    trainer = PyTorchTrainer(model, device)
                    
                    # Entraînement
                    history = trainer.train(
                        st.session_state['pytorch_data']['train_loader'],
                        st.session_state['pytorch_data']['test_loader'],
                        epochs=epochs,
                        lr=0.001,
                        patience=15
                    )
                    
                    st.session_state['lstm_model'] = model
                    st.session_state['trainer'] = trainer
                    
                    # Graphique de perte style sombre
                    fig, ax = plt.subplots(figsize=(10, 5), facecolor=COLORS['background'])
                    ax.set_facecolor(COLORS['card_bg'])
                    ax.plot(history['train_loss'], label='Train', color=COLORS['primary'], linewidth=2)
                    ax.plot(history['val_loss'], label='Validation', color=COLORS['secondary'], linewidth=2)
                    ax.set_title('Évolution de la perte - LSTM PyTorch', color=COLORS['text'])
                    ax.set_xlabel('Epochs', color=COLORS['text'])
                    ax.set_ylabel('Loss', color=COLORS['text'])
                    ax.tick_params(colors=COLORS['text'])
                    ax.legend()
                    ax.grid(True, alpha=0.1, color=COLORS['text'])
                    st.pyplot(fig)
                    
                    # Évaluation
                    model.eval()
                    with torch.no_grad():
                        X_test = st.session_state['pytorch_data']['X_test'].to(device)
                        y_test = st.session_state['pytorch_data']['y_test'].cpu().numpy()
                        
                        y_pred = model(X_test).cpu().numpy()
                        
                        y_test_inv = st.session_state['pytorch_data']['scaler'].inverse_transform(
                            y_test.reshape(-1, 1)
                        )
                        y_pred_inv = st.session_state['pytorch_data']['scaler'].inverse_transform(
                            y_pred.reshape(-1, 1)
                        )
                        
                        mse = mean_squared_error(y_test_inv, y_pred_inv)
                        mae = mean_absolute_error(y_test_inv, y_pred_inv)
                        
                        st.markdown(f"""
                        <div class="gradient-card">
                            <h4>📈 Performance LSTM PyTorch</h4>
                            <p>• MSE: {mse:.4f}</p>
                            <p>• MAE: {mae:.4f}</p>
                            <p>• Epochs: {len(history['train_loss'])}</p>
                            <p>• Final loss: {history['val_loss'][-1]:.6f}</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        with col2:
            if st.button("🎯 Entraîner NeuralProphet", use_container_width=True):
                model = train_neuralprophet(st.session_state['prophet_data'])
                st.session_state['prophet_model'] = model
                
                st.success("✅ Modèle NeuralProphet entraîné!")
                
                st.markdown(f"""
                <div class="gradient-card">
                    <h4>📈 NeuralProphet</h4>
                    <p>• Modèle entraîné avec succès</p>
                    <p>• Prêt pour les prédictions</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Étape 4: Prédictions
    if 'lstm_model' in st.session_state and 'prophet_model' in st.session_state:
        st.markdown("## 🔮 Étape 4: Prédictions")
        
        if st.button("Générer les prédictions", use_container_width=True):
            with st.spinner("Calcul des prédictions..."):
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                # Dernière séquence
                last_sequence = st.session_state['pytorch_data']['scaled_data'][
                    -st.session_state['pytorch_data']['look_back']:, 0
                ]
                
                # Prédictions LSTM
                lstm_pred = predict_future_lstm_pytorch(
                    st.session_state['lstm_model'],
                    last_sequence,
                    st.session_state['pytorch_data']['scaler'],
                    prediction_days,
                    device
                )
                
                # Prédictions NeuralProphet
                future = st.session_state['prophet_model'].make_future_dataframe(
                    st.session_state['prophet_data'],
                    periods=prediction_days
                )
                forecast = st.session_state['prophet_model'].predict(future)
                prophet_pred = forecast['yhat1'].values[-prediction_days:]
                
                # Dates futures
                last_date = st.session_state['data'].index[-1]
                future_dates = [last_date + timedelta(days=i+1) for i in range(prediction_days)]
                
                # Graphique sombre
                fig = plot_interactive_predictions(
                    st.session_state['data'],
                    lstm_pred,
                    prophet_pred,
                    future_dates
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Analyse
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="gradient-card">
                        <h4>🔥 LSTM PyTorch</h4>
                        <p>Premier: ${lstm_pred[0]:.2f}</p>
                        <p>Dernier: ${lstm_pred[-1]:.2f}</p>
                        <p>Variation: {((lstm_pred[-1]-lstm_pred[0])/lstm_pred[0]*100):.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="gradient-card">
                        <h4>📊 NeuralProphet</h4>
                        <p>Premier: ${prophet_pred[0]:.2f}</p>
                        <p>Dernier: ${prophet_pred[-1]:.2f}</p>
                        <p>Variation: {((prophet_pred[-1]-prophet_pred[0])/prophet_pred[0]*100):.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    avg_first = (lstm_pred[0] + prophet_pred[0]) / 2
                    avg_last = (lstm_pred[-1] + prophet_pred[-1]) / 2
                    
                    st.markdown(f"""
                    <div class="gradient-card">
                        <h4>🎯 Consensus</h4>
                        <p>Moyenne J1: ${avg_first:.2f}</p>
                        <p>Moyenne J{prediction_days}: ${avg_last:.2f}</p>
                        <p>Tendance: {"📈 HAUSSE" if avg_last > avg_first else "📉 BAISSE"}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Tableau style sombre automatique par Streamlit
                st.markdown("### 📅 Détail des prédictions")
                
                df_predictions = pd.DataFrame({
                    'Date': [d.strftime('%Y-%m-%d') for d in future_dates],
                    'LSTM (PyTorch)': [f"${x:.2f}" for x in lstm_pred],
                    'NeuralProphet': [f"${x:.2f}" for x in prophet_pred],
                    'Moyenne': [f"${(lstm_pred[i] + prophet_pred[i])/2:.2f}" for i in range(prediction_days)]
                })
                
                st.dataframe(
                    df_predictions,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Export
                csv = df_predictions.to_csv(index=False)
                st.download_button(
                    label="📥 Télécharger les prédictions (CSV)",
                    data=csv,
                    file_name=f"jnj_predictions_pytorch_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

if __name__ == '__main__':
    main()