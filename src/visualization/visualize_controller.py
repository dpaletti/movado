import plotly.express as px
import plotly.graph_objs as go
import pandas as pd


def visualize_mae(path: str):
    df = pd.read_csv(path)
    df = df.loc[df["Estimation"] == 1]
    fig = px.line(x=range(len(df["MAE"])), y=df["MAE"])
    fig.update_layout(
        title="Chained Estimator MAE",
        xaxis_title="Estimator calls",
        yaxis_title="MAE",
    )
    fig.show()


def visualize_threshold(path: str):
    df = pd.read_csv(path)
    df = df["Threshold"][0][0]
    print(df)
    fig = px.line(x=range(len(df)), y=df)
    fig.update_layout(
        title="Threshold",
        xaxis_title="Function Calls",
        yaxis_title="Threshold value",
    )
    # fig.show()


def visualize_calls(path: str):
    df = pd.read_csv(path)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(df["Estimation"]))),
            y=df["Estimation"],
            mode="markers",
            marker=dict(
                size=12,
                # I want the color to be green if
                # lower_limit ≤ y ≤ upper_limit
                # else red
                color=(df.Estimation == 1).astype("int"),
                colorscale=[[0, "red"], [1, "green"]],
            ),
        )
    )

    fig.update_layout(
        title="Exact and Approximated Calls",
        xaxis_title="Function calls",
        yaxis_title="IsApproximated",
    )
    fig.show()


visualize_mae("../../../resources/mean_distance_controller_sin_100.csv")
