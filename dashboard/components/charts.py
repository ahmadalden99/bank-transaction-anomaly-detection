from src.config import CHART_COLOR_SEQUENCE


def style_plotly_chart(fig, height=360, showlegend=True):
    fig.update_layout(
        template="plotly_white",
        colorway=CHART_COLOR_SEQUENCE,
        height=height,
        margin=dict(l=16, r=16, t=48, b=18),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            color="#44505D",
            size=12,
            family="Segoe UI, Arial, sans-serif",
        ),
        title=dict(
            font=dict(color="#17202A", size=14),
            x=0,
            y=0.97,
            xanchor="left",
            yanchor="top",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1,
            title=None,
            font=dict(size=11, color="#5E6975"),
            bgcolor="rgba(255,255,255,0)",
        ),
        hoverlabel=dict(
            bgcolor="#17202A",
            bordercolor="#17202A",
            font=dict(color="#FFFFFF", size=12),
        ),
        showlegend=showlegend,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#E8EBEF",
        zeroline=False,
        linecolor="#D6DAE0",
        tickfont=dict(size=11, color="#5E6975"),
        title_font=dict(size=12, color="#44505D"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E8EBEF",
        zeroline=False,
        linecolor="#D6DAE0",
        tickfont=dict(size=11, color="#5E6975"),
        title_font=dict(size=12, color="#44505D"),
    )
    fig.update_traces(marker_line_width=0, opacity=1, selector=dict(type="bar"))
    return fig
