from types import SimpleNamespace

from us_population_agent.agent.nodes.validate_sql import validate_sql

SETTINGS = SimpleNamespace(census_schema="CENSUS.PUBLIC", geography_table="APP.PUBLIC.V_GEOGRAPHY")
COLUMNS = [{"name": "B01001e1", "meaning": "Estimate > SEX BY AGE > Total"}]
GOOD = (
    'SELECT v.STATE_ABBR, SUM(d."B01001e1") AS TOTAL_POPULATION, COUNT(*) AS N_CBG '
    'FROM CENSUS.PUBLIC."2020_CBG_B01" d JOIN APP.PUBLIC.V_GEOGRAPHY v USING (CENSUS_BLOCK_GROUP) '
    "WHERE v.STATE_ABBR = 'TX' GROUP BY v.STATE_ABBR ORDER BY TOTAL_POPULATION DESC"
)


def check(sql: str) -> list[str]:
    return validate_sql(sql, settings=SETTINGS, data_table="2020_CBG_B01", columns=COLUMNS)


def test_valid_query_passes():
    assert check(GOOD) == []


def test_union_all_passes():
    first = GOOD.split(" ORDER BY")[0]
    assert check(f"{first} UNION ALL {first}") == []


def test_trailing_semicolon_passes():
    assert check(GOOD + ";") == []


def test_ddl_and_dml_rejected():
    assert check('DROP TABLE CENSUS.PUBLIC."2020_CBG_B01"')
    assert check('DELETE FROM CENSUS.PUBLIC."2020_CBG_B01"')
    assert check("CREATE TABLE t AS SELECT 1")


def test_multiple_statements_rejected():
    assert check(f"{GOOD}; DROP TABLE t")


def test_unparsable_rejected():
    assert check("SELEC FROM (")


def test_unknown_table_rejected():
    assert any("not allowed" in e for e in check(GOOD.replace("V_GEOGRAPHY", "OTHER")))
    assert any("not allowed" in e for e in check(GOOD.replace("CENSUS.PUBLIC.", "")))


def test_wrong_data_table_rejected():
    assert any("is not used" in e for e in check(GOOD.replace("2020_CBG_B01", "2020_CBG_B02")))


def test_unknown_column_rejected():
    assert any("not in the column list" in e for e in check(GOOD.replace("B01001e1", "B01001e9")))


def test_wrong_case_column_rejected():
    assert any("does not exist" in e for e in check(GOOD.replace("B01001e1", "B01001E1")))


def test_unquoted_column_rejected():
    assert any("quoted" in e for e in check(GOOD.replace('d."B01001e1"', "d.B01001e1")))


def test_column_without_alias_rejected():
    assert any("prefixed" in e for e in check(GOOD.replace('d."B01001e1"', '"B01001e1"')))
