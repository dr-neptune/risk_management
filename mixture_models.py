import numpy as np
from sklearn.mixture import GaussianMixture
import plotly.express as px

# generate data to be slightly overlapping
np.random.seed(42)
n_samples = 1000
mean_regime1 = [1, 1]
cov_regime1 = [[2, 0.5], [0.5, 2]]
mean_regime2 = [3, 3]
cov_regime2 = [[2, -0.5], [-0.5, 2]]

half = n_samples // 2
data_regime1 = np.random.multivariate_normal(mean_regime1, cov_regime1, half)
data_regime2 = np.random.multivariate_normal(mean_regime2, cov_regime2, half)
data = np.vstack([data_regime1, data_regime2])

# Fit a Gaussian mixture model to the overlapped 2D data
gmm = GaussianMixture(n_components=2, covariance_type='full')
gmm.fit(data)
labels = gmm.predict(data)

# Plotting using plotly with plotly_dark theme

fig = px.scatter(x=data[:, 0], y=data[:, 1], color=labels, labels={'x': 'Asset 1 Return', 'y': 'Asset 2 Return'})
fig.update_traces(marker=dict(size=5))
centers = gmm.means_
fig.add_trace(
    px.scatter(x=centers[:, 0], y=centers[:, 1], size=[30, 30], color_discrete_sequence=['red', 'red']).data[0]
)

fig.update_layout(title="Mixture of Normal Variance Distributions", template="plotly_dark")
fig.show()










import numpy as np
from sklearn.mixture import GaussianMixture
import plotly.express as px

# Step 1: Data Generation Function
def generate_data(num_clusters):
    np.random.seed(42)
    n_samples = 1000
    data_list = []

    # Define some base parameters and then add slight variations for each cluster
    base_mean = [1, 1]
    base_cov = [[2, 0.5], [0.5, 2]]

    for i in range(num_clusters):
        mean = [base_mean[0] + i, base_mean[1] + i]
        cov = [[base_cov[0][0] + i*0.2, base_cov[0][1]], [base_cov[1][0], base_cov[1][1] + i*0.2]]
        cluster_data = np.random.multivariate_normal(mean, cov, n_samples // num_clusters)
        data_list.append(cluster_data)

    return np.vstack(data_list)

# Step 2: Gaussian Mixture Model Function
def fit_gmm(data, num_components):
    gmm = GaussianMixture(n_components=num_components, covariance_type='full')
    gmm.fit(data)
    return gmm, gmm.predict(data)

# Step 3: Call both functions and plot the results using plotly for 5 clusters
data = generate_data(5)
gmm, labels = fit_gmm(data, 5)

fig = px.scatter(x=data[:, 0], y=data[:, 1], color=labels, labels={'x': 'Dimension 1', 'y': 'Dimension 2'})
fig.update_traces(marker=dict(size=5))
centers = gmm.means_
fig.add_trace(
    px.scatter(x=centers[:, 0], y=centers[:, 1], size=[30] * 5, color_discrete_sequence=['red'] * 5).data[0]
)

fig.update_layout(title="Mixture of Normal Variance Distributions with 5 Clusters", template="plotly_dark")
fig.show()




import yfinance as yf
import numpy as np
from sklearn.mixture import GaussianMixture
import plotly.express as px

# Fetch NFLX stock data for the mentioned eras
start_dates = ['1997-01-01', '2007-01-01', '2013-01-01']
end_dates = ['2007-01-01', '2013-01-01', '2023-01-01']

data_frames = [yf.download('NFLX', start=start_date, end=end_date)['Close'] for start_date, end_date in zip(start_dates, end_dates)]
data = np.concatenate([df.pct_change().dropna().values.reshape(-1, 1) for df in data_frames])

# Fit a GMM
gmm, labels = fit_gmm(data, 3)

# Plotting the results
fig = px.histogram(data, x=0, color=labels, labels={'x': 'Daily Return'}, nbins=100,
                   title="NFLX Stock Daily Returns with GMM Components")
fig.update_layout(template="plotly_dark")
fig.show()





import yfinance as yf
import numpy as np
from sklearn.mixture import GaussianMixture
import plotly.graph_objects as go

# Fetch NFLX stock data
nflx = yf.download('NFLX', start='1997-01-01', end='2023-01-01')['Close']
returns = nflx.pct_change().dropna().values.reshape(-1, 1)

# Fit a GMM to the returns
gmm, labels = fit_gmm(returns, 3)

# Plotting the results
fig = go.Figure()

# Color mapping based on the mixture component
colors = ['red', 'blue', 'green']

# Loop over each label and create a scatter trace for each one
for label in np.unique(labels):
    mask = labels == label
    fig.add_trace(go.Scatter(x=nflx.index[1:][mask], y=nflx.values[1:][mask], mode='lines',
                             line=dict(color=colors[label]), name=f'Component {label}'))

fig.update_layout(title='NFLX Stock Price with GMM Components', template="plotly_dark", xaxis_title='Date', yaxis_title='Price')
fig.show()




import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from hmmlearn import hmm

# Fetch NFLX stock data
nflx = yf.download('NFLX', start='1997-01-01', end='2023-01-01')['Close']
returns = nflx.pct_change().dropna().values.reshape(-1, 1)

# Fit a Hidden Markov Model to the returns
# Assuming 3 hidden states for simplicity
model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
model.fit(returns)
states = model.predict(returns)

# Plotting the results
fig = go.Figure()

# Color mapping based on the state
colors = ['red', 'blue', 'green']

# Loop over each state and create a scatter trace for each one
for state in np.unique(states):
    mask = states == state
    fig.add_trace(go.Scatter(x=nflx.index[1:][mask], y=nflx.values[1:][mask], mode='lines',
                             line=dict(color=colors[state]), name=f'State {state}'))

fig.update_layout(title='NFLX Stock Price with HMM States', template="plotly_dark", xaxis_title='Date', yaxis_title='Price')
fig.show()













import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from hmmlearn import hmm

# Fetch NFLX stock data
nflx = yf.download('NFLX', start='1997-01-01', end='2023-01-01')['Close']
returns = nflx.pct_change().dropna().values.reshape(-1, 1)

# Fit a Hidden Markov Model to the returns
# Assuming 3 hidden states for simplicity
model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
model.fit(returns)
states = model.predict(returns)

# Plotting the results
fig = go.Figure()

# Color mapping based on the state
colors = ['red', 'blue', 'green']

# Loop over each state and create a scatter trace for each one
for state in np.unique(states):
    mask = states == state
    fig.add_trace(go.Scatter(x=nflx.index[1:][mask], y=nflx.values[1:][mask], mode='markers',
                             marker=dict(color=colors[state], size=4), name=f'State {state}'))

fig.update_layout(title='NFLX Stock Price with HMM States (Scatterplot)', template="plotly_dark", xaxis_title='Date', yaxis_title='Price')
fig.show()
