import streamlit as st
import feedparser
from pytrends.request import TrendReq
import pandas as pd
import altair as alt
import re
import html
from datetime import datetime, timedelta
from dateutil import parser as dateparser
import pytz
import plotly.express as px
import time, random
import os, pickle
from fpdf import FPDF
import unicodedata
from io import BytesIO
from collections import Counter
from pathlib import Path

# RSS Feed URLs
rss_feeds = [
    "https://www.google.com/alerts/feeds/00243988972911864076/8161066467534929256",
    "https://www.google.com/alerts/feeds/00243988972911864076/12932915266585500269",
    "https://www.google.com/alerts/feeds/00243988972911864076/17382340201875275568",
    "https://www.google.com/alerts/feeds/00243988972911864076/14856341163007996523",
    "https://www.google.com/alerts/feeds/00243988972911864076/1773085610071443605",
    "https://www.google.com/alerts/feeds/00243988972911864076/9843631178445806850",
    "https://www.google.com/alerts/feeds/00243988972911864076/5548442157361180539",
    "https://www.google.com/alerts/feeds/00243988972911864076/1209388039159001449",
    "https://www.google.com/alerts/feeds/00243988972911864076/12398715252804110702",
    "https://www.google.com/alerts/feeds/00243988972911864076/16349084675887028901",
    "https://www.google.com/alerts/feeds/00243988972911864076/10449138946122118959",
    "https://www.google.com/alerts/feeds/00243988972911864076/11774744901597160664",
    "https://www.google.com/alerts/feeds/00243988972911864076/9782522591869606635",
    "https://www.google.com/alerts/feeds/00243988972911864076/5628816019690244625",
    "https://www.google.com/alerts/feeds/00243988972911864076/3920679245701179674",
    "https://www.google.com/alerts/feeds/00243988972911864076/15841105531779883613"
]

trend_keywords = [
    "Fireproofing",
    "Monokote",
    "UL Rating",
    "Fireproofing Contractor",
    "UL Design",
    "CAFCO",
    "Isolatek"
]

# US States list for filtering (abbreviations)
us_states = ["All","AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"]

# Mapping from state name to abbreviation for choropleth
state_name_to_abbrev = {
    'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA','Colorado':'CO','Connecticut':'CT','Delaware':'DE','District of Columbia':'DC','Florida':'FL','Georgia':'GA','Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA','Kansas':'KS','Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD','Massachusetts':'MA','Michigan':'MI','Minnesota':'MN','Mississippi':'MS','Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV','New Hampshire':'NH','New Jersey':'NJ','New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND','Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC','South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA','Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY'
}

# --- Helper to detect states in text ---
state_abbrevs = [s for s in us_states if s != "All"]

def get_states_in_text(text: str):
    found = set()
    upper = text.upper()
    for ab in state_abbrevs:
        if f" {ab} " in f" {upper} ":
            found.add(ab)
    return found

@st.cache_data(ttl=1800)
def get_feed_items(url):
    feed = feedparser.parse(url)
    items = []
    for entry in feed.entries:
        raw_title = entry.title
        clean_title = html.unescape(re.sub(r'<[^>]+>', '', raw_title))
        summary_html = entry.summary if 'summary' in entry else ''
        items.append({
            "title": clean_title,
            "link": entry.link,
            "summary": summary_html,
            "published": entry.published if 'published' in entry else ''
        })
    return items

def matches_filters(item, selected_state, require_dollar, keywords, last_days):
    text = f"{item['title']} {item.get('summary','')}"
    # State filter
    if selected_state != "All":
        pattern = r'\b' + re.escape(selected_state) + r'\b'
        if not re.search(pattern, text, re.IGNORECASE):
            return False
    # Dollar amount filter
    if require_dollar and not re.search(r'\$\s?\d', text):
        return False
    # Keyword filter (any keyword)
    if keywords:
        if not any(re.search(rf"{re.escape(k)}", text, re.IGNORECASE) for k in keywords):
            return False
    # Date filter
    if last_days is not None and last_days > 0 and item.get("published"):
        try:
            published_dt = dateparser.parse(item["published"])
            if published_dt is not None and datetime.utcnow() - published_dt > timedelta(days=last_days):
                return False
        except Exception:
            pass
    return True

def highlight_text(text, keywords):
    if not keywords:
        return text
    def _repl(match):
        return f"<mark>{match.group(0)}</mark>"
    for kw in keywords:
        text = re.sub(rf"({re.escape(kw)})", _repl, text, flags=re.IGNORECASE)
    return text

# --- Badge rules and helpers ---
BADGE_RULES = {
    "Airport": [r"\bairport\b", r"\bconstruction\b|\bproject\b"],
    "Casino": [r"\bcasino\b", r"\bconstruction\b|\bproject\b"],
    "College Stadium": [r"(college stadium|university arena|athletic facility)", r"(project|expansion)"],
    "Data Center": [r"\bdata center\b", r"\bproject\b"],
    "GC Awarded": [r"general contractor", r"awarded"],
    "Groundbreaking": [r"groundbreaking ceremony", r"construction"],
    "High Rise": [r"high rise", r"construction|project"],
    "Hospital": [r"hospital", r"construction|project"],
    "Megaproject": [r"\bmega ?project\b|\bmegaproject\b"],
    "Mega Construction": [r"\bmega ?project\b|\bmegaproject\b", r"construction"],
    "Project Announced": [r"project announced", r"construction"],
    "Semiconductor": [r"semiconductor", r"construction|project"],
    "Tower": [r"\btower\b", r"construction|project"],
    "Pro Sports Venue": [r"(nfl|nba|mlb|nhl|mls)", r"(stadium|arena)", r"(construction|renovation)"],
    "Stadium Project": [r"(sports stadium|arena|ballpark|coliseum|athletic facility)", r"(construction|renovation|expansion|groundbreaking|project)"],
    "University Project": [r"\b(university)\b", r"(construction|project)"],
}

def get_custom_badges(text: str):
    text = text.lower()
    matched = []
    for badge, patterns in BADGE_RULES.items():
        if all(re.search(p, text) for p in patterns):
            matched.append(badge)
    return matched

def render_badges(badges):
    return " ".join([
        f"<span style='background:#FF5722;color:white;padding:4px 8px;border-radius:5px;margin-right:4px;font-size:0.8rem'>{b}</span>"
        for b in badges
    ])

def show_rss_feeds(selected_feeds, selected_state, require_dollar, max_items, keywords, last_days):
    st.header("📰 Intelligence Feed")
    state_counts = Counter()
    for url in selected_feeds:
        feed_items = get_feed_items(url)
        filtered_items = [item for item in feed_items if matches_filters(item, selected_state, require_dollar, keywords, last_days)]
        for idx, item in enumerate(filtered_items[:max_items]):
            # count state mentions
            states_in = get_states_in_text(item['title'] + ' ' + item.get('summary',''))
            state_counts.update(states_in)

            title_html = highlight_text(item["title"], keywords)
            text_for_badges = f"{item['title']} {item.get('summary','')}"
            badges = get_custom_badges(text_for_badges)

            with st.expander(title_html):
                st.write(item.get("published",""))
                if badges:
                    st.markdown(render_badges(badges), unsafe_allow_html=True)
                summary_html = highlight_text(item.get("summary",""), keywords)
                st.markdown(summary_html, unsafe_allow_html=True)
                st.markdown(f"[🔗 Read more]({item['link']})")

                # PDF download button
                summary_plain = re.sub(r"<[^>]+>", "", item.get("summary",""))
                pdf_bytes = generate_pdf(item["title"], item.get("published",""), summary_plain, item["link"])
                file_name = sanitize_filename(item["title"])
                st.download_button(
                    label="📄 Download PDF",
                    data=pdf_bytes,
                    file_name=file_name,
                    mime="application/pdf",
                    key=f"pdf_{idx}_{item['link']}"
                )
            st.markdown("---")
    # save counts for other tabs
    st.session_state['alert_state_counts'] = dict(state_counts)

# --- Google Trends helper ---
def safe_build_payload(pytrends, kw_list, timeframe, geo=''):
    """Retry build_payload with backoff to avoid 429"""
    for attempt in range(3):
        try:
            pytrends.build_payload(kw_list, timeframe=timeframe, geo=geo)
            return True
        except Exception as e:
            if '429' in str(e) and attempt < 2:
                time.sleep(random.uniform(12, 18))
                continue
            raise

@st.cache_data(ttl=3600)
def fetch_trends_df(keywords, timeframe):
    """Fetch Google Trends data in batches of ≤5 keywords to avoid 400 errors."""
    pytrends = TrendReq()
    frames = []
    for i in range(0, len(keywords), 5):
        batch = keywords[i:i+5]
        try:
            if not safe_build_payload(pytrends, batch, timeframe):
                continue
            df_part = pytrends.interest_over_time()
            if not df_part.empty:
                frames.append(df_part.drop(columns=['isPartial']))
        except Exception as e:
            st.warning(f"Trend fetch failed for {', '.join(batch)}: {e}")
    if frames:
        df_all = pd.concat(frames, axis=1)
        df_all = df_all.loc[:,~df_all.columns.duplicated()]
        return df_all
    return pd.DataFrame()

# --- Disk cache fetch helper ---
def fetch_and_cache_trends(keywords, timeframe):
    filename = f"trends_cache_{'_'.join([k.replace(' ','_') for k in keywords])}_{timeframe}.pkl"
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    pytrends = TrendReq()
    if not safe_build_payload(pytrends, keywords, timeframe):
        return pd.DataFrame()

    df = pytrends.interest_over_time()
    if not df.empty:
        with open(filename, 'wb') as f:
            pickle.dump(df, f)
    return df

# --- Heat map helper ---
@st.cache_data(ttl=3600)
def fetch_heat_df(keyword, timeframe):
    """Return DataFrame with columns Code, Interest for US states with simple retry/backoff."""
    for attempt in range(3):
        try:
            pytrends = TrendReq()
            if not safe_build_payload(pytrends, [keyword], timeframe, geo='US'):
                continue
            reg_df = pytrends.interest_by_region(resolution='REGION', inc_low_vol=True)
            if reg_df.empty:
                return pd.DataFrame()
            # Ensure state column named 'State'
            reg_df = reg_df.rename_axis('State').reset_index()
            if keyword in reg_df.columns:
                reg_df = reg_df.rename(columns={keyword:'Interest'})
            else:
                reg_df.columns = ['State','Interest']
            reg_df['Code'] = reg_df['State'].map(state_name_to_abbrev)
            reg_df = reg_df.dropna(subset=['Code'])
            return reg_df[['Code','Interest']]
        except Exception as e:
            if '429' in str(e) and attempt < 2:
                sleep_for = random.uniform(12,18)
                time.sleep(sleep_for)
                continue
            st.warning(f"Heat map fetch failed: {e}")
            return pd.DataFrame()

def show_google_trends(*_args):
    st.header("📈 Google Trends Tracker")

    timeframe = st.selectbox("Timeframe", ['today 1-m', 'today 3-m', 'today 12-m', 'today 5-y'])

    if st.button("Fetch Trends", key="fetch_trends_btn_v3"):
        df = fetch_and_cache_trends(trend_keywords, timeframe)

        if not df.empty:
            df = df.drop(columns=['isPartial'])
            df = df.reset_index().melt('date', var_name='Keyword', value_name='Interest')

            chart = alt.Chart(df).mark_line().encode(
                x='date:T',
                y='Interest:Q',
                color='Keyword:N'
            ).properties(width=800, height=400)

            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("No data available or blocked by Google.")
    else:
        st.info("Click the button to fetch fresh trend data.")

    # --- Alerts Heat Map ---
    if 'alert_state_counts' in st.session_state and st.session_state['alert_state_counts']:
        st.subheader("Alerts Mentions Heat Map")
        heat_counts = st.session_state['alert_state_counts']
        df_heat = pd.DataFrame({'Code': list(heat_counts.keys()), 'Count': list(heat_counts.values())})
        fig2 = px.choropleth(df_heat, locations='Code', locationmode='USA-states', color='Count', scope='usa', color_continuous_scale='Reds', labels={'Count':'Mentions'})
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Heat map will appear after Alerts data with state mentions is loaded.")

# --- PDF helper ---
@st.cache_data
def generate_pdf(title: str, published: str, summary: str, link: str) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    def to_latin1(text):
        return unicodedata.normalize('NFKD', text).encode('latin-1','ignore').decode('latin-1')

    # Header
    pdf.set_font("Arial", 'B', 18)
    pdf.multi_cell(0, 10, to_latin1("FP Intelligence System"), align='C')
    pdf.ln(4)

    # Title
    pdf.set_font("Arial", size=16)
    pdf.multi_cell(0, 10, to_latin1(title))
    pdf.ln()
    pdf.set_font("Arial", size=10)
    if published:
        pdf.multi_cell(0, 8, to_latin1(f"Published: {published}"))
        pdf.ln(1)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, to_latin1(summary))
    pdf.ln()
    pdf.set_text_color(0,0,255)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0,8, to_latin1(link))
    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    return pdf_bytes

def sanitize_filename(text: str) -> str:
    return re.sub(r"[^\w\-_ ]", "", text)[:60] + ".pdf"

def _find_logo():
    asset_dir = Path(__file__).parent / "assets"
    # preferred exact names
    for ext in ("png","jpg","jpeg","gif"):
        p = asset_dir / f"fireproof_logo.{ext}"
        if p.exists():
            return p
    # fallback: any file containing 'logo'
    for p in asset_dir.glob("*logo*.*"):
        if p.is_file():
            return p
    return None

# --- Summary metrics helper ---
def compute_summary_metrics(feeds, selected_state, require_dollar, keywords, last_days):
    """Return mentions today and top term among trend keywords."""
    mentions_today = 0
    term_counts = Counter()
    today = datetime.utcnow().date()
    for url in feeds:
        feed_items = get_feed_items(url)
        for item in feed_items:
            if not matches_filters(item, selected_state, require_dollar, keywords, last_days):
                continue
            # mentions today
            if item.get("published"):
                try:
                    published_dt = dateparser.parse(item["published"])
                    if published_dt.date() == today:
                        mentions_today += 1
                except Exception:
                    pass
            # count keyword occurrences in title
            for kw in trend_keywords:
                if re.search(rf"\b{re.escape(kw)}\b", item["title"], re.IGNORECASE):
                    term_counts[kw] += 1
    top_term = term_counts.most_common(1)[0][0] if term_counts else "-"
    return mentions_today, top_term

st.set_page_config(
    page_title="Fireproofing Intelligence Dashboard",
    page_icon="🔥",
    layout="wide"
)

# Branding Header
logo_path = _find_logo()

st.markdown(
    """
    <h1 style='margin-bottom:0;'>Fireproof SWARM Dashboard</h1>
    <p style='margin-top:0; font-size: 1.1rem; color: gray;'>Live Alerts + Trends Monitoring for Fireproofing Intelligence</p>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    if logo_path is not None:
        st.image(str(logo_path), width=300)
    else:
        st.markdown("### Fireproof SWARM")
    st.markdown("#### Powered by Fireproof SWARM")
    with st.expander("Filter Options", expanded=False):
        selected_state = st.selectbox("Filter by US State", us_states)
        require_dollar = st.checkbox("Require $ value", value=False)
        keyword_input = st.text_input("Keyword search (comma separated)")
        keywords = [k.strip() for k in keyword_input.split(',') if k.strip()]
        last_days = st.slider("Show last N days", 1, 30, 7)
        max_items = st.slider("Max articles per feed", 1, 50, 15)
    st.markdown(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

selected_feeds = rss_feeds  # always use all feeds; hide URLs from UI

# --- Summary Metrics ---
with st.spinner("Scanning Google Alerts feeds..."):
    mentions_today, top_term = compute_summary_metrics(selected_feeds, selected_state, require_dollar, keywords, last_days)

mc1, mc2 = st.columns(2)
mc1.metric(label="Mentions Today", value=str(mentions_today))
mc2.metric(label="Top Term", value=top_term)

# --- Tabs ---
tabs = st.tabs(["Alerts", "Trends (Coming Soon)"])

with tabs[0]:
    show_rss_feeds(selected_feeds, selected_state, require_dollar, max_items, keywords, last_days)

with tabs[1]:
    st.header("🚧 Trends - Coming Soon")
    st.info("Google Trends visuals are temporarily disabled due to rate-limit issues. Stay tuned!")
