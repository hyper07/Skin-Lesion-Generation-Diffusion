import streamlit as st


def render_footer() -> None:
    # Spacer so fixed footer doesn't overlap content
    st.markdown('<div class="footer-spacer"></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="footer-bleed">
            <div class="footer-inner">
                <p class="brand">AML Analysis Platform</p>
                <p>Developed by Columbia University SPS Capstone Team in collaboration with Société Générale</p>
                <p style="opacity:0.75;">© 2025 All Rights Reserved</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


