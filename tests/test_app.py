from streamlit.testing.v1 import AppTest
from src.config import ROOT


def test_dashboard_loads_without_exceptions():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=20)
    app.run()
    assert not app.exception
    assert app.title[0].value == "🏀 Hidden Court"
    assert app.metric[0].label == "Candidatos"


def test_empty_filters_do_not_show_unrelated_players():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=20).run()
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert app.metric[0].value == "0"
    assert app.metric[1].value == "—"
    assert not app.selectbox
    assert app.info
