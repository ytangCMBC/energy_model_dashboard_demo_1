# beb_all_panels_dashboard.py

import streamlit as st

from beb_blk_dashboard import render_block_panel
from beb_onroute_dashboard import render_onroute_panel
from beb_trip_dashboard import render_trip_panel

st.set_page_config(
    page_title="Energy Model – Multi-Level Dashboard",
    layout="wide",
)

st.title("Energy Model – Multi-Level Dashboard")

with st.sidebar:
    st.markdown("## View")
    panel = st.radio(
        "Select panel",
        [             
            "Route / Trip level",
            "Block level (Depot-only)",
            "Block level (On-route charging)"
  
        ],
        index=0,
    )

if panel == "Block level (Depot-only)":
    render_block_panel()
elif panel == "Block level (On-route charging)":
    render_onroute_panel()
else:
    render_trip_panel()
