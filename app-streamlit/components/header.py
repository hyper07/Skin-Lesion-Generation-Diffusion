import streamlit as st


def render_header(active: str | None = None) -> None:
    """Render a fixed, full-width header bar with logo. Menu is handled by st.navigation."""
    # Full-width header bar
    st.markdown('<div class="app-header"><div class="app-header-inner">', unsafe_allow_html=True)
    st.image("images/Logo_SP.png", width=180)

    st.markdown('</div></div>', unsafe_allow_html=True)

    # Spacer so body content doesn't hide under fixed header
    st.markdown('<div class="header-spacer"></div>', unsafe_allow_html=True)


