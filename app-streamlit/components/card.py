import streamlit.components.v1 as components
import streamlit as st
import os


def render_nav_card(icon: str, title: str, description: str, action_key: str, page_path: str, height: int = 220) -> None:
    """
    Render a clickable navigation card using components.html
    
    Args:
        icon: Emoji or icon to display
        title: Card title
        description: Card description text
        action_key: Unique action key (for styling/identification)
        page_path: Path to the page to navigate to (e.g., "pages/1_EDA.py")
        height: Height of the card component in pixels
    """
    # Map action keys to button labels
    button_labels = {
        'nav_eda': 'Navigate to EDA',
        'nav_analytics': 'Navigate to Analytics',
        'nav_sar': 'Navigate to SAR'
    }
    button_label = button_labels.get(action_key, 'Navigate')
    
    card_html = f"""
    <style>
        .nav-card-link {{
            text-decoration: none;
            display: block;
            color: inherit;
        }}
        .nav-card {{
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border: 2px solid #e0e6ed;
            padding: 2.5rem 2rem;
            margin: 1rem 0;
            border-radius: 16px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.08);
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        .nav-card:hover {{
            transform: translateY(-6px);
            box-shadow: 0 12px 32px rgba(10,53,82,0.2);
            border-color: #0a3552;
            background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%);
        }}
        .nav-card .nav-icon {{
            font-size: 3.5rem;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, #0a3552 0%, #0a3552 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .nav-card h4 {{
            color: #2c3e50;
            margin: 0 0 0.75rem 0;
            font-size: 1.4rem;
            font-weight: 700;
        }}
        .nav-card p {{
            color: #6c757d;
            margin: 0 0 1.5rem 0;
            font-size: 1rem;
            line-height: 1.6;
        }}
    </style>
    <a href="#" class="nav-card-link" onclick="
        var buttons = Array.from(window.parent.document.querySelectorAll('button'));
        var btn = buttons.find(b => (b.innerText || '').trim() === '{button_label}');
        if (btn) {{ btn.click(); }}
        return false;
    ">
        <div class="nav-card">
            <div class="nav-icon">{icon}</div>
            <h4>{title}</h4>
            <p>{description}</p>
        </div>
    </a>
    """
    
    components.html(card_html, height=height)


def setup_card_message_listener() -> None:
    """
    Hide navigation buttons since cards are now clickable via anchor tags.
    This should be called once on the page that uses nav cards.
    """
    import streamlit as st
    st.markdown("""
    <style>
        /* Hide navigation buttons since cards are clickable */
        button[data-testid*="nav_eda_btn"],
        button[data-testid*="nav_analytics_btn"],
        button[data-testid*="nav_sar_btn"] {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)

