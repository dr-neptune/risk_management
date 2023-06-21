import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.templates.default = "plotly_dark"


def plot_multiple_time_series(series_dict, opacity_dict=None, file_name=None, title=None):
    # Create a subplot
    fig = make_subplots()

    # Add a trace for each series
    for label, series in series_dict.items():
        # Get the opacity for this series if provided, else default to 1
        opacity = opacity_dict.get(label, 1) if opacity_dict else 1

        fig.add_trace(
            go.Scatter(x=series.index, y=series, name=label, opacity=opacity)
        )

    # Set the title and axis labels
    if title is None:
        title = " ~ ".join(series_dict.keys())

    fig.update_layout(
        title_text=title,
        showlegend=True
    )

    if file_name is not None:
        fig.write_html(f'figures/{file_name}.html')

    # Show the plot
    fig.show()
