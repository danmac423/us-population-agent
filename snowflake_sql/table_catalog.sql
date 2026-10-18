-- 1) Table-level corpus: one row per ACS table, built from the original metadata
CREATE OR REPLACE TABLE CENSUS_APP_DB.PUBLIC.TABLE_CATALOG AS
WITH fields AS (
    SELECT
        TABLE_NUMBER AS TABLE_ID,                                -- e.g. B01001
        TRY_TO_NUMBER(REGEXP_SUBSTR(src.TABLE_ID, 'e([0-9]+)$', 1, 1, 'e', 1)) AS COL_NUM,   -- e.g. 49
        '2020_CBG_' || SUBSTR(TABLE_NUMBER, 1, 3) AS DATA_TABLE_NAME,       -- e.g. 2020_CBG_B01
        TABLE_TITLE,
        TABLE_TOPICS,
        TABLE_UNIVERSE,
        ARRAY_CONSTRUCT_COMPACT(
            FIELD_LEVEL_1, FIELD_LEVEL_2, FIELD_LEVEL_3, FIELD_LEVEL_4, FIELD_LEVEL_5,
            FIELD_LEVEL_6, FIELD_LEVEL_7, FIELD_LEVEL_8,
            "FIELD_LEVELl_9",                                    -- Column name is misspelled in source
            FIELD_LEVEL_10
        ) AS LEVELS
    FROM US_OPEN_CENSUS_DATA_NEIGHBORHOOD_INSIGHTS_FREE_DATASET.PUBLIC."2020_METADATA_CBG_FIELD_DESCRIPTIONS" src
    WHERE src.TABLE_ID LIKE '%e%'                                     -- Take only the estimates (the actual data, discarding the 'm' margins of errors)
      AND TABLE_NUMBER NOT LIKE 'B99%'                                -- Exclude table B99 (Imputed/Allocated answers - data quality)
),
tables AS (
    SELECT TABLE_ID,
           ANY_VALUE(DATA_TABLE_NAME) AS DATA_TABLE_NAME,
           ANY_VALUE(TABLE_TITLE)     AS TABLE_TITLE,
           ANY_VALUE(TABLE_TOPICS)    AS TABLE_TOPICS,
           ANY_VALUE(TABLE_UNIVERSE)  AS TABLE_UNIVERSE,
           COUNT(*)                   AS N_COLUMNS
    FROM fields
    GROUP BY TABLE_ID
),
-- Every distinct value at each depth below "Estimate > TITLE > Universe", skipping 'Total',
-- remembering the first column it appears in so values keep census order
values_by_depth AS (
    SELECT f.TABLE_ID, l.INDEX AS RAW_DEPTH, l.VALUE::STRING AS VAL, MIN(f.COL_NUM) AS FIRST_COL
    FROM fields f, LATERAL FLATTEN(f.LEVELS) l
    WHERE l.INDEX >= 3 AND l.VALUE::STRING <> 'Total'
    GROUP BY 1, 2, 3
),
level_values AS (
    SELECT TABLE_ID,
           DENSE_RANK() OVER (PARTITION BY TABLE_ID ORDER BY RAW_DEPTH) AS DEPTH,   -- renumber 1, 2, ...
           LISTAGG(VAL, ', ') WITHIN GROUP (ORDER BY FIRST_COL) AS VALS
    FROM values_by_depth
    GROUP BY TABLE_ID, RAW_DEPTH
),
breakdowns AS (
    SELECT TABLE_ID,
           LISTAGG('Level ' || DEPTH || ': ' || VALS, ' | ') WITHIN GROUP (ORDER BY DEPTH) AS BREAKDOWNS
    FROM level_values
    GROUP BY TABLE_ID
)
SELECT t.TABLE_ID, t.DATA_TABLE_NAME, t.TABLE_TITLE, t.N_COLUMNS,
       b.BREAKDOWNS,                                             -- NULL for single-column (Total-only) tables
       'Table: ' || COALESCE(t.TABLE_TITLE, '') ||
       ' | Topics: ' || COALESCE(t.TABLE_TOPICS, '') ||
       ' | Universe: ' || COALESCE(t.TABLE_UNIVERSE, '') ||
       ' | Breakdowns: ' || COALESCE(b.BREAKDOWNS, '') AS SEARCH_TEXT
FROM tables t
LEFT JOIN breakdowns b USING (TABLE_ID);

-- 2) Search service
CREATE OR REPLACE CORTEX SEARCH SERVICE CENSUS_APP_DB.PUBLIC.CENSUS_TABLE_SEARCH
  ON SEARCH_TEXT ATTRIBUTES DATA_TABLE_NAME
  WAREHOUSE = US_POPULATION_WH TARGET_LAG = '1 day'
  EMBEDDING_MODEL = 'snowflake-arctic-embed-l-v2.0'
AS SELECT TABLE_ID, DATA_TABLE_NAME, TABLE_TITLE, SEARCH_TEXT FROM CENSUS_APP_DB.PUBLIC.TABLE_CATALOG;