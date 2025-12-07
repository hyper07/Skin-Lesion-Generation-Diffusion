## Parameters
The `option_menu` function accepts the following parameters:
- menu_title (required): the title of the menu; pass None to hide the title
- options (required): list of (string) options to display in the menu; set an option to "---" if you want to insert a section separator
- default_index (optional, default=0): the index of the selected option by default
- menu_icon (optional, default="menu-up"): name of the [bootstrap-icon](https://icons.getbootstrap.com/) to be used for the menu title
- icons (optional, default=["caret-right"]): list of [bootstrap-icon](https://icons.getbootstrap.com/) names to be used for each option; its length should be equal to the length of options
- orientation (optional, default="vertical"): "vertical" or "horizontal"; whether to display the menu vertically or horizontally
- styles (optional, default=None): A dictionary containing the CSS definitions for most HTML elements in the menu, including:
    * "container": the container div of the entire menu
    * "menu-title": the &lt;a> element containing the menu title
    * "menu-icon": the icon next to the menu title
    * "nav": the &lt;ul> containing "nav-link"
    * "nav-item": the &lt;li> element containing "nav-link"
    * "nav-link": the &lt;a> element containing the text of each option
    * "nav-link-selected": the &lt;a> element containing the text of the selected option
    * "icon": the icon next to each option
    * "separator": the &lt;hr> element separating the options
- manual_select: Pass to manually change the menu item selection. 
The function returns the (string) option currently selected
- on_change: A callback that will happen when the selection changes. The callback function should accept one argument "key". You can use it to fetch the value of the menu (see [example 5](#examples))



### Manual Selection
This option was added to allow the user to manually move to a specific option in the menu. This could be useful when the user wants to move to another option automatically after finishing with one option (for example, if settings are approved, then move back to the main option).

To use this option, you need to pass the index of the desired option as `manual_select`. **Notice**: This option behaves like a button. This means that you should only pass `manual_select` once when you want to select the option, and not keep it as a constant value in your menu creation call (see example below).


## Examples
```python
import streamlit as st
from streamlit_option_menu import option_menu

# 1. as sidebar menu
with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'Settings'], 
        icons=['house', 'gear'], menu_icon="cast", default_index=1)
    selected

# 2. horizontal menu
    selected2 = option_menu(None, ["Home", "Upload", "Tasks", 'Settings'], 
        icons=['house', 'cloud-upload', "list-task", 'gear'], 
        menu_icon="cast", default_index=0, orientation="horizontal")
    selected2

# 3. CSS style definitions
    selected3 = option_menu(None, ["Home", "Upload",  "Tasks", 'Settings'], 
        icons=['house', 'cloud-upload', "list-task", 'gear'], 
        menu_icon="cast", default_index=0, orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "25px"}, 
            "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "green"},
        }
)

# 4. Manual item selection
if st.session_state.get('switch_button', False):
    st.session_state['menu_option'] = (st.session_state.get('menu_option', 0) + 1) % 4
    manual_select = st.session_state['menu_option']
else:
    manual_select = None
    
selected4 = option_menu(None, ["Home", "Upload", "Tasks", 'Settings'], 
    icons=['house', 'cloud-upload', "list-task", 'gear'], 
    orientation="horizontal", manual_select=manual_select, key='menu_4')
st.button(f"Move to Next {st.session_state.get('menu_option', 1)}", key='switch_button')
selected4

# 5. Add on_change callback
def on_change(key):
    selection = st.session_state[key]
    st.write(f"Selection changed to {selection}")
    
selected5 = option_menu(None, ["Home", "Upload", "Tasks", 'Settings'],
                        icons=['house', 'cloud-upload', "list-task", 'gear'],
                        on_change=on_change, key='menu_5', orientation="horizontal")
selected5
```

---

## Database Configuration (`database_config.py`)

The `database_config` module provides utilities to connect to the PostgreSQL database using environment variables or docker-compose.yml configuration.

### Configuration Priority

The module reads database configuration in the following order:
1. **Environment variables** (preferred) - Set in docker-compose.yml or .env file
2. **docker-compose.yml** (fallback) - Parsed automatically if env vars not available

### Environment Variables

The following environment variables are automatically set in docker-compose.yml for the Streamlit service:

```yaml
POSTGRES_USER=admin
POSTGRES_PASSWORD=PassW0rd
POSTGRES_DB=db
POSTGRES_HOST=capstone-postgres
POSTGRES_PORT=5432
DATABASE_URL=postgresql://admin:PassW0rd@capstone-postgres:5432/db
```

### Local Development

For local development, you can create a `.env` file in the `app-streamlit` directory:

```bash
# Option 1: Use DATABASE_URL (recommended)
DATABASE_URL=postgresql://admin:PassW0rd@localhost:55432/db

# Option 2: Use individual variables
POSTGRES_USER=admin
POSTGRES_PASSWORD=PassW0rd
POSTGRES_DB=db
POSTGRES_HOST=localhost
POSTGRES_PORT=55432
```

### Usage Examples

```python
from functions.database_config import (
    execute_query,
    get_database_url,
    get_all_tables,
    test_connection
)

# Get database connection URL
db_url = get_database_url()
print(f"Database URL: {db_url}")

# Test connection
if test_connection():
    print("✅ Database connection successful!")
else:
    print("❌ Database connection failed")

# Execute a query and get DataFrame
df = execute_query("SELECT * FROM transactions LIMIT 10")
print(df)

# Get all tables
tables = get_all_tables()
print(tables)

# Get table schema
schema = get_table_schema("transactions")
print(schema)
```

### Key Functions

- `get_database_url(use_host=False)` - Get SQLAlchemy connection URL
- `get_db_connection_params(use_host=False)` - Get connection parameters as dict
- `create_db_engine(use_host=False)` - Create SQLAlchemy engine
- `execute_query(query, use_host=False, return_df=True)` - Execute SQL and return DataFrame
- `test_connection(use_host=False)` - Test database connectivity
- `get_all_tables(use_host=False)` - List all tables
- `get_table_schema(table_name, use_host=False)` - Get table schema
- `get_table_row_count(table_name, use_host=False)` - Get row count

### Notes

- When running in Docker, the module automatically uses the container network hostname (`capstone-postgres`)
- When running locally, use `use_host=True` to connect via `localhost:55432`
- The module automatically detects Docker environment and adjusts connection settings accordingly