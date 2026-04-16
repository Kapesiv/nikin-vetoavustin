import streamlit as st
import numpy as np
import pandas as pd
import requests
from math import factorial, exp
from itertools import product
from typing import Optional, List
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Apufunktiot nimien siivoamiseen
# ---------------------------------------------------------------------------
def _clean_team_name(raw: str) -> str:
    """Siivoa joukkueen nimi duplikaateista."""
    for sep in ["Allsvenskan", "Sweden", "FormLast"]:
        if sep in raw:
            raw = raw[: raw.index(sep)]
    n = len(raw)
    for half in range(n // 2, 3, -1):
        if raw[:half] == raw[half : half * 2]:
            return raw[:half].strip()
    return raw.strip()


def _clean_bolldata_name(raw: str) -> str:
    """Siivoa Bolldatan duplikaattinimi (esim. 'AIKAIK' -> 'AIK')."""
    raw = raw.strip()
    n = len(raw)
    for half in range(n // 2, 2, -1):
        if raw[:half] == raw[half : half * 2]:
            return raw[:half].strip()
    return raw


# ---------------------------------------------------------------------------
# 1) FootyStats-skraapaus
# ---------------------------------------------------------------------------
FOOTYSTATS_XG_URL = "https://footystats.org/sweden/allsvenskan/xg"


@st.cache_data(ttl=900)
def scrape_footystats_xg() -> Optional[List[dict]]:
    """Skraappaa joukkueiden xG-tilastot FootyStats-sivulta."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        }
        resp = requests.get(FOOTYSTATS_XG_URL, headers=headers, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")
        table = soup.find("table")
        if not table:
            return None

        teams = []
        for row in table.find_all("tr"):
            tds = row.find_all("td")
            if len(tds) < 49:
                continue
            if "position" not in (tds[0].get("class") or []):
                continue

            raw_name = tds[2].get_text(strip=True)
            name = _clean_team_name(raw_name)

            mp = int(float(tds[43].get_text(strip=True)))
            xg_total = float(tds[44].get_text(strip=True))
            xga_total = float(tds[45].get_text(strip=True))
            gf_total = float(tds[47].get_text(strip=True))
            ga_total = float(tds[48].get_text(strip=True))

            teams.append({
                "name": name,
                "mp": mp,
                "xg_per_game": round(xg_total / mp, 3) if mp > 0 else 0,
                "xga_per_game": round(xga_total / mp, 3) if mp > 0 else 0,
                "gf_per_game": round(gf_total / mp, 2) if mp > 0 else 0,
                "ga_per_game": round(ga_total / mp, 2) if mp > 0 else 0,
                "xg_total": xg_total,
                "xga_total": xga_total,
            })

        return teams if teams else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 2) Bolldata.se-skraapaus
# ---------------------------------------------------------------------------
BOLLDATA_URL = "https://bolldata.se/lagdata"


@st.cache_data(ttl=900)
def scrape_bolldata() -> Optional[List[dict]]:
    """Skraappaa xG & xGA -taulukko Bolldata.se:stä."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        }
        resp = requests.get(BOLLDATA_URL, headers=headers, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")
        tables = soup.find_all("table")

        # Etsi xG & xGA -taulukko (sarakkeet: LAG, SM, GM, xG, ±, IM, xGA, ±)
        xg_table = None
        for t in tables:
            header_row = t.find("tr")
            if not header_row:
                continue
            ths = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]
            if "xG" in ths and "xGA" in ths and "GM" in ths:
                xg_table = t
                break

        if not xg_table:
            return None

        teams = []
        for row in xg_table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            # Suodata tyhjät rivit
            cells = [c for c in cells if c]
            if len(cells) < 8:
                continue

            name = _clean_bolldata_name(cells[0])
            mp = int(cells[1])
            gf = int(cells[2])
            xg = float(cells[3].replace(",", "."))
            ga = int(cells[5])
            xga = float(cells[6].replace(",", "."))

            teams.append({
                "name": name,
                "mp": mp,
                "xg_per_game": round(xg / mp, 3) if mp > 0 else 0,
                "xga_per_game": round(xga / mp, 3) if mp > 0 else 0,
                "gf_per_game": round(gf / mp, 2) if mp > 0 else 0,
                "ga_per_game": round(ga / mp, 2) if mp > 0 else 0,
                "xg_total": xg,
                "xga_total": xga,
            })

        return teams if teams else None
    except Exception as e:
        st.error(f"Bolldata-virhe: {e}")
        return None


# ---------------------------------------------------------------------------
# Poisson-malli
# ---------------------------------------------------------------------------
MAX_GOALS = 8


def poisson_pmf(lam: float, k: int) -> float:
    return exp(-lam) * lam**k / factorial(k)


def build_score_matrix(lambda_home: float, lambda_away: float) -> np.ndarray:
    matrix = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
    for h, a in product(range(MAX_GOALS + 1), repeat=2):
        matrix[h, a] = poisson_pmf(lambda_home, h) * poisson_pmf(lambda_away, a)
    return matrix


def match_probabilities(matrix: np.ndarray) -> dict:
    home_win = np.sum(np.tril(matrix, -1).T)
    draw = np.sum(np.diag(matrix))
    away_win = np.sum(np.triu(matrix, 1).T)
    return {"home": home_win, "draw": draw, "away": away_win}


def prob_to_odds(p: float) -> float:
    return round(1 / p, 2) if p > 0 else float("inf")


def expected_lambda(
    attack_xg: float,
    defence_xg_against: float,
    league_avg_goals: float,
) -> float:
    if league_avg_goals == 0:
        return 0.0
    attack_strength = attack_xg / league_avg_goals
    defence_weakness = defence_xg_against / league_avg_goals
    return attack_strength * defence_weakness * league_avg_goals


# ---------------------------------------------------------------------------
# Streamlit-käyttöliittymä
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Nikin Vetoavustin", layout="wide")
st.title("Nikin Vetoavustin")

# Sivupalkki
st.sidebar.header("Asetukset")
source = st.sidebar.radio(
    "Datalähde",
    ["Bolldata.se", "FootyStats.org", "Keskiarvo (molemmat)"],
    index=2,
)

# Lataa data molemmista lähteistä
with st.sidebar:
    with st.spinner("Haetaan dataa..."):
        footystats_data = scrape_footystats_xg()
        bolldata_data = scrape_bolldata()

status_parts = []
if bolldata_data:
    status_parts.append(f"Bolldata: {len(bolldata_data)} joukkuetta")
if footystats_data:
    status_parts.append(f"FootyStats: {len(footystats_data)} joukkuetta")

if status_parts:
    st.sidebar.success(" | ".join(status_parts))
else:
    st.sidebar.error("Molemmat lähteet epäonnistuivat")


# Yhdistä data valitun lähteen mukaan
def merge_sources(
    bd: Optional[List[dict]],
    fs: Optional[List[dict]],
    mode: str,
) -> List[dict]:
    """Yhdistä tai valitse datalähde."""
    if mode == "Bolldata.se":
        return bd or fs or []
    if mode == "FootyStats.org":
        return fs or bd or []

    # Keskiarvo: yhdistä joukkueet nimen perusteella
    if not bd and not fs:
        return []
    if not bd:
        return fs
    if not fs:
        return bd

    bd_map = {t["name"]: t for t in bd}
    fs_map = {t["name"]: t for t in fs}
    all_names = set(bd_map.keys()) | set(fs_map.keys())

    merged = []
    for name in all_names:
        b = bd_map.get(name)
        f = fs_map.get(name)
        if b and f:
            merged.append({
                "name": name,
                "mp": max(b["mp"], f["mp"]),
                "xg_per_game": round((b["xg_per_game"] + f["xg_per_game"]) / 2, 3),
                "xga_per_game": round((b["xga_per_game"] + f["xga_per_game"]) / 2, 3),
                "gf_per_game": round((b["gf_per_game"] + f["gf_per_game"]) / 2, 2),
                "ga_per_game": round((b["ga_per_game"] + f["ga_per_game"]) / 2, 2),
                "xg_total": round((b["xg_total"] + f["xg_total"]) / 2, 2),
                "xga_total": round((b["xga_total"] + f["xga_total"]) / 2, 2),
                "bolldata_xg": b["xg_per_game"],
                "footystats_xg": f["xg_per_game"],
                "bolldata_xga": b["xga_per_game"],
                "footystats_xga": f["xga_per_game"],
            })
        else:
            t = b or f
            merged.append({**t, "bolldata_xg": None, "footystats_xg": None,
                           "bolldata_xga": None, "footystats_xga": None})
    return merged


teams_data = merge_sources(bolldata_data, footystats_data, source)

if not teams_data:
    st.error("Dataa ei saatavilla.")
    st.stop()

team_names = sorted([t["name"] for t in teams_data])
team_map = {t["name"]: t for t in teams_data}

# Liigan keskiarvot
league_xg_for = np.mean([t["xg_per_game"] for t in teams_data])
league_xg_against = np.mean([t["xga_per_game"] for t in teams_data])
league_avg = (league_xg_for + league_xg_against) / 2

# ---------------------------------------------------------------------------
# Välilehdet
# ---------------------------------------------------------------------------
tab_match, tab_table, tab_value = st.tabs(
    ["Otteluennuste", "xG-sarjataulukko", "Arvovetolaskuri"]
)

# ========================  VÄLILEHTI 1: Otteluennuste  =====================
with tab_match:
    st.subheader("Otteluennuste (Poisson-malli)")
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Kotijoukkue", team_names, index=0)
    with col2:
        away_default = 1 if len(team_names) > 1 else 0
        away_team = st.selectbox("Vierasjoukkue", team_names, index=away_default)

    if home_team == away_team:
        st.warning("Valitse kaksi eri joukkuetta.")
    else:
        ht = team_map[home_team]
        at = team_map[away_team]

        lambda_home = expected_lambda(ht["xg_per_game"], at["xga_per_game"], league_avg)
        lambda_away = expected_lambda(at["xg_per_game"], ht["xga_per_game"], league_avg)

        matrix = build_score_matrix(lambda_home, lambda_away)
        probs = match_probabilities(matrix)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Koti xG", f"{lambda_home:.2f}")
        m2.metric("Vieras xG", f"{lambda_away:.2f}")
        m3.metric(f"{home_team} voitto", f"{probs['home']:.1%}")
        m4.metric("Tasapeli", f"{probs['draw']:.1%}")
        m5.metric(f"{away_team} voitto", f"{probs['away']:.1%}")

        st.markdown("**Todelliset kertoimet (ilman marginaalia)**")
        o1, o2, o3 = st.columns(3)
        o1.metric("1", prob_to_odds(probs["home"]))
        o2.metric("X", prob_to_odds(probs["draw"]))
        o3.metric("2", prob_to_odds(probs["away"]))

        with st.expander("Tulosmatriisi (todennäköisyydet)", expanded=False):
            score_df = pd.DataFrame(
                matrix[:6, :6],
                index=[f"Koti {i}" for i in range(6)],
                columns=[f"Vieras {j}" for j in range(6)],
            )
            st.dataframe(score_df.style.format("{:.1%}").background_gradient(cmap="YlOrRd"))

        with st.expander("Yli / Alle maalimäärä", expanded=False):
            for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
                over_p = sum(
                    matrix[h, a]
                    for h in range(MAX_GOALS + 1)
                    for a in range(MAX_GOALS + 1)
                    if h + a > line
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.write(f"Yli {line}")
                c2.write(f"{over_p:.1%}")
                c3.write(f"Alle {line}")
                c4.write(f"{1 - over_p:.1%}")

        with st.expander("Molemmat tekevät maalin (MTMS)", expanded=False):
            btts_yes = sum(
                matrix[h, a]
                for h in range(1, MAX_GOALS + 1)
                for a in range(1, MAX_GOALS + 1)
            )
            b1, b2 = st.columns(2)
            b1.metric("MTMS Kyllä", f"{btts_yes:.1%}")
            b2.metric("MTMS Ei", f"{1 - btts_yes:.1%}")


# ========================  VÄLILEHTI 2: xG-sarjataulukko  ==================
with tab_table:
    st.subheader("xG-yhteenveto – Kaikki joukkueet")
    rows = []
    for t in teams_data:
        row = {
            "Joukkue": t["name"],
            "Ottelut": t.get("mp", 0),
            "xG/ottelu": t["xg_per_game"],
            "xGA/ottelu": t["xga_per_game"],
            "xG-ero/ottelu": round(t["xg_per_game"] - t["xga_per_game"], 2),
            "Maaleja/ottelu": t["gf_per_game"],
            "Päästetty/ottelu": t["ga_per_game"],
        }
        # Näytä molemmat lähteet jos keskiarvo-tilassa
        if source == "Keskiarvo (molemmat)":
            bd_xg = t.get("bolldata_xg")
            fs_xg = t.get("footystats_xg")
            if bd_xg is not None and fs_xg is not None:
                row["BD xG"] = bd_xg
                row["FS xG"] = fs_xg
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("xG-ero/ottelu", ascending=False).reset_index(drop=True)
    df.index += 1

    fmt = {
        "xG/ottelu": "{:.2f}",
        "xGA/ottelu": "{:.2f}",
        "xG-ero/ottelu": "{:.2f}",
        "Maaleja/ottelu": "{:.2f}",
        "Päästetty/ottelu": "{:.2f}",
    }
    if "BD xG" in df.columns:
        fmt["BD xG"] = "{:.2f}"
        fmt["FS xG"] = "{:.2f}"

    st.dataframe(
        df.style.format(fmt).background_gradient(subset=["xG-ero/ottelu"], cmap="RdYlGn"),
        use_container_width=True,
    )


# ========================  VÄLILEHTI 3: Arvovetolaskuri  ===================
with tab_value:
    st.subheader("Arvovetolaskuri")
    st.markdown(
        "Vertaa mallin todellisia kertoimia vedonlyöjän kertoimiin löytääksesi arvovedot. "
        "**Etu = (Todellinen todennäköisyys × Vedonlyöjän kerroin) − 1**"
    )

    col_home, col_away = st.columns(2)
    with col_home:
        vb_home = st.selectbox("Kotijoukkue", team_names, index=0, key="vb_home")
    with col_away:
        vb_away_idx = 1 if len(team_names) > 1 else 0
        vb_away = st.selectbox("Vierasjoukkue", team_names, index=vb_away_idx, key="vb_away")

    if vb_home == vb_away:
        st.warning("Valitse kaksi eri joukkuetta.")
    else:
        ht = team_map[vb_home]
        at = team_map[vb_away]

        lh = expected_lambda(ht["xg_per_game"], at["xga_per_game"], league_avg)
        la = expected_lambda(at["xg_per_game"], ht["xga_per_game"], league_avg)
        mat = build_score_matrix(lh, la)
        pr = match_probabilities(mat)

        st.markdown("**Mallin todelliset kertoimet**")
        f1, f2, f3 = st.columns(3)
        f1.metric(f"1 ({vb_home})", prob_to_odds(pr["home"]))
        f2.metric("X", prob_to_odds(pr["draw"]))
        f3.metric(f"2 ({vb_away})", prob_to_odds(pr["away"]))

        st.markdown("**Syötä vedonlyöjän kertoimet**")
        b1, b2, b3 = st.columns(3)
        with b1:
            bk_home = st.number_input("Kerroin 1", min_value=1.01, value=2.50, step=0.05, key="bk1")
        with b2:
            bk_draw = st.number_input("Kerroin X", min_value=1.01, value=3.30, step=0.05, key="bkx")
        with b3:
            bk_away = st.number_input("Kerroin 2", min_value=1.01, value=2.80, step=0.05, key="bk2")

        stake = st.number_input("Panos per veto (yksikköä)", min_value=0.0, value=10.0, step=1.0)

        results = []
        for label, fair_p, bk_odds in [
            (f"1 – {vb_home}", pr["home"], bk_home),
            ("X – Tasapeli", pr["draw"], bk_draw),
            (f"2 – {vb_away}", pr["away"], bk_away),
        ]:
            edge = fair_p * bk_odds - 1
            ev = edge * stake
            kelly = (fair_p * bk_odds - 1) / (bk_odds - 1) if bk_odds > 1 else 0
            results.append({
                "Lopputulos": label,
                "Mallin tod.näk.": f"{fair_p:.1%}",
                "Todell. kerroin": prob_to_odds(fair_p),
                "Vedonlyöjän kerroin": bk_odds,
                "Etu": f"{edge:+.1%}",
                "Odotusarvo (yks.)": round(ev, 2),
                "Kelly %": f"{max(kelly, 0):.1%}",
                "Arvoveto?": "KYLLÄ" if edge > 0 else "ei",
            })

        res_df = pd.DataFrame(results)

        def highlight_value(row):
            if row["Arvoveto?"] == "KYLLÄ":
                return ["background-color: #d4edda"] * len(row)
            return [""] * len(row)

        st.dataframe(
            res_df.style.apply(highlight_value, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        value_bets = [r for r in results if r["Arvoveto?"] == "KYLLÄ"]
        if value_bets:
            st.success(
                f"Löytyi {len(value_bets)} arvoveto(a): "
                + ", ".join(r["Lopputulos"] for r in value_bets)
            )
        else:
            st.info("Arvovetoja ei löytynyt nykyisillä vedonlyöjän kertoimilla.")

# Alatunniste
st.divider()
st.caption(
    "Data: Bolldata.se & FootyStats.org (automaattinen päivitys 15 min välein). "
    "Malli: Riippumaton Poisson-jakauma xG-keskiarvojen pohjalta."
)
